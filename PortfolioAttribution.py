### This function is designed to calcualte the portfolio attribution in which it should receive the orders, historical prices, holidays and nav data
### The mappings are designed to transform the input data in a data the function can read, and the user should change the values according to his input file.

import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

## Importing Data:
path = "C:/Users/joaop/OneDrive/Documentos/Estudos/Planilhas e Códigos/Códigos/Portfolio/Attribution/Attribution.xlsx"
orders = pd.read_excel(path, sheet_name='Ordens').dropna(how="all")
history = pd.read_excel(path, sheet_name='Historico',index_col=0)
holidays = pd.read_excel(path, sheet_name='Feriados', parse_dates=True, header=None).iloc[:,0]
nav_data = pd.read_excel(path, sheet_name='Cotas', parse_dates=True)

#Inputs:
start_date = ""
end_date = ""
attribution_type = "Gross"

#Mapping inputs from the column names
orders_mapping = {
    "DATE": "Data",
    "SECURITY TYPE": "Classe",
    "BOOK": "Book",
    "SECURITY": "Ativo",
    "SIDE": "Direção",
    "QUANTITY": "Tamanho",
    "PRICE": "Preço",
    "TOTAL ORDER VALUE": "Total da Ordem",
    "CURRENCY": "Dólar/Euro"
}

orders_value_mapping = {
    'SIDE': {'V': 'Sell', 'C': 'Buy'},
}

nav_mapping = {
    "DATE": "Data",
    "NAV": "PL - Sirius",
    "QUANTITY": "Quantidade de Cotas",
    "NAVPS": "Cota"
}

cash_mapping = {
    "DATE": "Data",
    "CASH": "Caixa"
}

benchmark_mapping = {
    "DATE": "Data",
    "BENCHMARK": "CDI"
}

gains_mapping = {
    "Ações Americanas": "Change_Currency",
    "Ações Europeias": "Change_Currency", 
    "Futuros": "Exposure",
    "Moedas": "Change",
    "Ações Listadas na B3": "Change"
}

gains_currency_value_mapping = {
    "Ações Americanas": history['PTAX'],
    "Ações Europeias": history['EUR/BRL']
}


def portfolio_attribution(orders, nav, history, holidays, attribution_type="Gross", cash=None, benchmark=None, start_date=None, end_date=None):
    ## Mapping Data: (Code starts here)
    def create_dataframe_from_mapping(data_df, column_mapping):
        # Create an empty dataframe based on the target column names
        target_columns = list(column_mapping.keys())
        new_df = pd.DataFrame(columns=target_columns)

        # Loop through the dictionary to extract and rename columns
        for target_column, source_column in column_mapping.items():
            if source_column in data_df.columns:
                new_df[target_column] = data_df[source_column]

        return new_df

    def replace_values_in_columns(data_df, column_value_mappings):
        # Create a copy of the DataFrame to avoid modifying the original
        modified_df = data_df.copy()

        for column_name, value_mapping in column_value_mappings.items():
            # Replace values in the specified column based on the mapping
            modified_df[column_name] = modified_df[column_name].map(value_mapping).fillna(modified_df[column_name])

        return modified_df

    def calculate_gains(df, type, *args):
        if type == 'Exposure':
            return df
        elif type == 'Change_Currency':
            if len(args) != 1:
                raise ValueError("The 'Change_Currency' type requires one additional argument (currency_df).")
            currency_df = args[0]
            return df * currency_df - df.shift(1) * currency_df.shift(1)
        elif type == 'Change':
            return df - df.shift(1)
        else:
            raise ValueError(f"Type '{type}' is not supported.")
    
    #Mapping inputs from the column values

    orders = replace_values_in_columns(create_dataframe_from_mapping(orders, orders_mapping), orders_value_mapping).set_index('DATE')
    nav = create_dataframe_from_mapping(nav, nav_mapping).set_index('DATE')
    cash = create_dataframe_from_mapping(cash, cash_mapping).set_index('DATE')
    benchmark = create_dataframe_from_mapping(benchmark, benchmark_mapping).set_index('DATE')

    # Set default start and end dates if not provided
    start_date = start_date if start_date is not None else nav.index[0].strftime("%Y-%m-%d")
    end_date = end_date if end_date is not None else nav.index[-1].strftime("%Y-%m-%d")

    # Calculate benchmark returns
    benchmark_returns = benchmark.pct_change().fillna(method='ffill')

    # Calculate adjusted quantities in orders
    orders['ADJUSTED_QUANTITY'] = orders['QUANTITY'] * orders['SIDE'].apply(lambda x: -1 if x == 'Sell' else 1)

    # Create a table with positions
    positions = (
        pd.pivot_table(orders, values=['ADJUSTED_QUANTITY'], columns=['BOOK','SECURITY'], index=orders.index)
        .reindex(pd.bdate_range(orders.index[0]-dt.timedelta(1), end_date, freq='C', holidays=holidays.to_list()))
        .fillna(0)
        .cumsum()
        .droplevel(0, 1)
        .loc[start_date:]
    )

    # Create a table with securities and their types
    classes = orders[['SECURITY','SECURITY TYPE']].drop_duplicates(subset='SECURITY').set_index('SECURITY')

    # Create a dataframe with financial exposure per day
    exposures = pd.DataFrame(index=positions.index,columns=positions.columns)

    for book in positions.columns.levels[0].tolist():
        prices_data = history[positions[book].columns.intersection(history.columns)].fillna(0)
        exposures[book] = positions[book] * prices_data

    # Create a dataframe with daily gains for each type of stock
    gains = pd.DataFrame(index=exposures.loc[:end_date].index, columns=exposures.columns)

    for book, security in gains.columns.to_flat_index():
        security_type = classes.loc[security, 'SECURITY TYPE']
        calculation_type = gains_mapping.get(security_type, "Change")
    
        if calculation_type == "Change_Currency":
            currency_df = gains_currency_value_mapping.get(security_type, 1)
        else:
            currency_df = 1
    
        gains[(book, security)] = calculate_gains(exposures[(book, security)], calculation_type, currency_df)  

    if attribution_type == "Net":
        gains -= exposures * benchmark_returns.values
    elif attribution_type != "Gross":
        pass

    # Create a dataframe with the value spent/received from every operation and add it to the gains dataframe
    operations = (
        pd.pivot_table(orders, values=['TOTAL ORDER VALUE'], columns=['BOOK', 'SECURITY'], index=orders.index)
        .reindex(pd.bdate_range(orders.index[0] - dt.timedelta(1), gains.index[-1], freq='C', holidays=holidays.to_list()))
        .fillna(0)
    )['TOTAL ORDER VALUE']

    gains_adjusted = gains.copy()

    for (book, security) in gains.columns.to_flat_index():
        security_type = classes.loc[security, 'SECURITY TYPE']
        mapping_type = gains_mapping.get(security_type, "Change")

        if mapping_type != "Exposure":
            gains_adjusted[(book, security)] += operations[(book, security)]


    # Calculate the contribution of the value in cash + costs
    gains_adjusted['Cash'] = (((nav['NAV'] - nav['NAV'].shift(1))) - gains_adjusted.sum(axis=1))

    # Define dates of the attribution
    def get_first_business_day_in_month(date, holidays): #Primeiro dia útil do mês da end_date
        date = datetime.strptime(date, '%Y-%m-%d')
        start_date = datetime(date.year, date.month, 1)
        while start_date.weekday() >= 5 or start_date in holidays:
            start_date += timedelta(days=1)
        return start_date

    def get_last_business_day_in_previous_month(date, holidays): #Último dia útil do mês anterior a end_date
        date = datetime.strptime(date, '%Y-%m-%d')
        first_of_month = datetime(date.year, date.month, 1)
        end_of_last_month = first_of_month - timedelta(days=1)
        while end_of_last_month.weekday() >= 5 or end_of_last_month in holidays:
            end_of_last_month -= timedelta(days=1)
        return end_of_last_month

    def get_first_day_of_year(date, nav_index, holidays): #Primeiro dia útil do ano da end_date (ou segundo dia o fundo)
        date = datetime.strptime(date, '%Y-%m-%d')
        start_date = datetime(date.year, 1, 1)
        if start_date.weekday() >= 5 or start_date in holidays:
            start_date += timedelta(days=1)
        start_date = max(start_date, nav_index[1])
        while start_date.weekday() >= 5 or start_date in holidays:
            start_date += timedelta(days=1)
        return start_date

    def get_last_business_day_of_previous_year(date, nav_index, holidays): #Último dia anterior do ano anterior a end_date (ou primeiro dia do fundo)
        date = datetime.strptime(date, '%Y-%m-%d')
        start_date = datetime(date.year - 1, 12, 31)
        start_date = max(start_date, nav_index[0])
        while start_date.weekday() >= 5 or start_date in holidays:
            start_date -= timedelta(days=1)
        return start_date

    start_month = get_first_business_day_in_month(end_date, holidays)
    end_month = get_last_business_day_in_previous_month(end_date, holidays)
    start_year = get_first_day_of_year(end_date, nav.index, holidays)
    end_year = get_last_business_day_of_previous_year(end_date, nav.index, holidays)
    day_before = nav.index[-2]

    # Stack the dataframe to calculate the contribution
    gains_adjusted_stacked = (
        gains_adjusted.stack(level=0)
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'DATE', 0: 'GAIN'})
        .set_index('DATE')
    )

    gains_adjusted_stacked.loc[end_date:,'Contribution Day'] = ((gains_adjusted_stacked.loc[end_date:, 'GAIN'] / nav.loc[day_before,'NAV']))
    gains_adjusted_stacked.loc[end_date:,'Gain Day'] = gains_adjusted_stacked.loc[end_date:, 'GAIN']

    # Divide all gains of the month by the last NAV of the previous month
    gains_adjusted_stacked.loc[start_month:,'Contribution Month'] = (gains_adjusted_stacked.loc[start_month:, 'GAIN'] / nav.loc[end_month,'NAV'])
    gains_adjusted_stacked.loc[start_month:,'Gain Month'] = gains_adjusted_stacked.loc[start_month:, 'GAIN']

    # Divide all gains of the year by the last NAV of the year before, or by the initial NAV (whichever comes first)
    gains_adjusted_stacked.loc[start_year:,'Contribution Year'] = (gains_adjusted_stacked.loc[start_year:, 'GAIN'] / nav.loc[end_year,'NAV'])
    gains_adjusted_stacked.loc[start_year:,'Gain Year'] = gains_adjusted_stacked.loc[start_year:, 'GAIN']

    gains_adjusted_stacked = gains_adjusted_stacked.fillna(0)

    # Consolidate Attribution
    attribution = pd.pivot_table(gains_adjusted_stacked, values=['Contribution Day', 'Gain Day', 'Contribution Month', 'Gain Month', 'Contribution Year', 'Gain Year'], columns=['BOOK','SECURITY'], aggfunc=sum).T
    subtotal = attribution.groupby(level=0).sum()
    subtotal.index = pd.MultiIndex.from_tuples([(x, 'Total') for x in subtotal.index])
    attribution = attribution.drop('Cash')
    attribution = pd.concat([attribution, subtotal])

    # Format Adjustments
    format_percentage = lambda x: '{:.2%}'.format(x)
    format_currency = lambda x: "R${:,.2f}".format(x)
    replace_decimal = lambda x: x.replace('.', ';').replace(',', '.').replace(';', ',')

    # Apply formatting using applymap
    attribution[['Contribution Day', 'Contribution Month', 'Contribution Year']] = attribution[['Contribution Day', 'Contribution Month', 'Contribution Year']].applymap(format_percentage)
    attribution[['Gain Day', 'Gain Month', 'Gain Year']] = attribution[['Gain Day', 'Gain Month', 'Gain Year']].applymap(format_currency)
    attribution = attribution.applymap(replace_decimal)

    return(attribution)


attribution = portfolio_attribution(orders, nav_data, history, holidays, "Gross", nav_data, nav_data)

print(attribution)