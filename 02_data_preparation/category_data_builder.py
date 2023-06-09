def build_category_data(category):

    import numpy as np
    import pandas as pd

    #Join datasets and select category entries
    transactions = pd.read_csv("../00_original_data/transaction_data.csv")
    products = pd.read_csv("../00_original_data/product.csv")
    transactions = pd.merge(transactions, products, how = "left", on = "PRODUCT_ID")
    category_data = transactions[transactions["COMMODITY_DESC"].isin(category)]
    
    #Per product and household, define first purchase event
    (category_data.sort_values(by=["household_key", "PRODUCT_ID", "DAY", "TRANS_TIME"], 
        inplace=True))
    grouped = category_data.groupby(["household_key", "PRODUCT_ID"])
    first_purchase = grouped.first().reset_index()
    category_data_2 = (pd.merge(category_data, first_purchase, on=["household_key", "PRODUCT_ID", "WEEK_NO", "DAY", "TRANS_TIME"], how="left"))
    
    # create "first_purchase" column, True if there is no NaN in any of the columns, False otherwise
    category_data_2["first_purchase"] = ~category_data_2.isnull().any(axis=1)
    
    #Prepare clean dataset
    category_data_2.drop(labels = 
                     ["BASKET_ID_y","QUANTITY_y", "SALES_VALUE_y", "STORE_ID_y", "RETAIL_DISC_y", "COUPON_DISC_y",
                      "COUPON_MATCH_DISC_y", "MANUFACTURER_y", "DEPARTMENT_y", "BRAND_y", "COMMODITY_DESC_y", 
                      "SUB_COMMODITY_DESC_y", "CURR_SIZE_OF_PRODUCT_y"],
                     axis = 1, inplace = True)
    category_data_2.rename(columns = 
                       {"BASKET_ID_x": "BASKET_ID", "QUANTITY_x" : "QUANTITY", "SALES_VALUE_x" : "SALES_VALUE",
                        "STORE_ID_x" : "STORE_ID", "RETAIL_DISC_x" : "RETAIL_DISC", "COUPON_DISC_x" : "COUPON_DISC",
                      "COUPON_MATCH_DISC_x" : "COUPON_MATCH_DISC", "MANUFACTURER_x" : "MANUFACTURER",
                        "DEPARTMENT_x" : "DEPARTMENT", "BRAND_x" : "BRAND", "COMMODITY_DESC_x" : "COMMODITY", 
                      "SUB_COMMODITY_DESC_x" : "SUB_COMMODITY", "CURR_SIZE_OF_PRODUCT_x" : "CURRENT_SIZE_OF_PRODUCT"},
                       inplace = True)
    
    #calculate regular shelf price w/o discounts
    category_data_2["shelf_price"] = (category_data_2["SALES_VALUE"] - category_data_2["RETAIL_DISC"] - 
                                  category_data_2["COUPON_MATCH_DISC"]) / category_data_2["QUANTITY"]
    #calculate paid shelf price with discounts
    category_data_2["paid_price"] = (category_data_2["SALES_VALUE"] + category_data_2["COUPON_DISC"]) / category_data_2["QUANTITY"]
    #calculate overall discount
    category_data_2["pct_disc"] = (category_data_2["shelf_price"]-category_data_2["paid_price"]) / category_data_2["shelf_price"]
    #calculate discount in percentage points versus regualar shelf price due to retailer loyalty card 
    category_data_2["pct_retail_disc"] = (-category_data_2["RETAIL_DISC"] / category_data_2["QUANTITY"]) / category_data_2["shelf_price"]
    #calculate discount in percentage points versus regualar shelf price due to manufacturer coupons 
    category_data_2["pct_coupon_disc"] = ((-category_data_2["COUPON_DISC"]-category_data_2["COUPON_MATCH_DISC"]) /
                                      category_data_2["QUANTITY"] ) / category_data_2["shelf_price"]
    
    category_data = category_data_2[(category_data_2["shelf_price"] > 0) 
                                  & (category_data_2["QUANTITY"] > 0) & (category_data_2["paid_price"] > 0)]

    #Add in causal / marketing data
    causal = pd.read_parquet("../00_original_data/causal_data.parquet")
    causal_category = pd.merge(causal, products[["PRODUCT_ID", "COMMODITY_DESC"]], how = "left", on = "PRODUCT_ID")
    causal_category = causal_category[causal_category.COMMODITY_DESC.isin(category)]
    category_data = pd.merge(category_data, causal_category, how = "left", on = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
    
    #remove duplicates
    category_data.drop_duplicates(
        subset= ["household_key", "BASKET_ID", "STORE_ID", "DAY", "TRANS_TIME", "PRODUCT_ID"], keep = "last", inplace= True)
    
    #dummy variables for display
    dummies_display = pd.get_dummies(category_data["display"], prefix = "display")
    category_data = pd.concat((category_data, dummies_display), axis = 1)
    #dummy variables for mailer
    dummies_mailer = pd.get_dummies(category_data["mailer"], prefix = "mailer")
    category_data = pd.concat((category_data, dummies_mailer), axis = 1)
    category_data.drop(labels = ["display", "mailer", "display_0", "mailer_0"], axis = 1, inplace = True)
    
    #Add in household demographics
    hhdemogs = pd.read_csv("../00_original_data/hh_demographic.csv")
    category_data = pd.merge(category_data, hhdemogs, how = "left", on="household_key")
    
    #dummify household demogs
    dummies_marital_status = pd.get_dummies(category_data["MARITAL_STATUS_CODE"], prefix = "marital_status")
    dummies_homeowner = pd.get_dummies(category_data["HOMEOWNER_DESC"], prefix = "homeowner")
    dummies_hhcomp = pd.get_dummies(category_data["HH_COMP_DESC"], prefix = "hhcomp")
    dummies_kidcat = pd.get_dummies(category_data["KID_CATEGORY_DESC"], prefix = "kid_category")
    category_data = pd.concat((category_data, dummies_marital_status), axis = 1)
    category_data = pd.concat((category_data, dummies_homeowner), axis = 1)
    category_data = pd.concat((category_data, dummies_hhcomp), axis = 1)
    category_data = pd.concat((category_data, dummies_kidcat), axis = 1)
        
    #ordinal features age, income and household size
    map_age = {"19-24":1, "25-34":2, "35-44":3, "45-54":4, "55-64":5, "65+":6}
    category_data= category_data.assign(age_ordinal = category_data.AGE_DESC.map(map_age))
    map_income = {"Under 15K":10, "15-24K":19.5, "25-34K":29.5, "35-49K":39.5, "50-74K":62, "75-99K":87, "100-124K":112, 
              "125-149K":137, "150-174K":162, "175-199K":187, "200-249K":224.5, "250K+":250}
    category_data = category_data.assign(income_ordinal = category_data.INCOME_DESC.map(map_income))
    map_hhsize = {"1":1, "2":2, "3":3, "4":4, "5+":5}
    category_data = category_data.assign(hhsize_ordinal = category_data.HOUSEHOLD_SIZE_DESC.map(map_hhsize))
    
    #dummy features age, income and household size
    dummies_age = pd.get_dummies(category_data["AGE_DESC"], prefix = "age")
    dummies_income = pd.get_dummies(category_data["INCOME_DESC"], prefix = "income")
    dummies_hhsize = pd.get_dummies(category_data["HOUSEHOLD_SIZE_DESC"], prefix = "hhsize")
    category_data = pd.concat((category_data, dummies_age), axis = 1)
    category_data = pd.concat((category_data, dummies_income), axis = 1)
    category_data = pd.concat((category_data, dummies_hhsize), axis = 1)
    
    category_data.drop(labels = ["AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC", 
                             "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC", 
                             "marital_status_U", "homeowner_Unknown", "hhcomp_Unknown"],
                   axis = 1, inplace = True)
    
    #Add in campaign data
    campaign_desc = pd.read_csv("../00_original_data/campaign_desc.csv")
    coupon = pd.read_csv("../00_original_data/coupon.csv")
    campaign_table = pd.read_csv("../00_original_data/campaign_table.csv")
    products = products[products["COMMODITY_DESC"].isin(category)]
    campaign_data = pd.merge(products, coupon, how = "left", on = "PRODUCT_ID")
    campaign_data = pd.merge(campaign_data, campaign_desc, how = "left", on = "CAMPAIGN")
    campaign_data = pd.merge(campaign_data, campaign_table, how = "left", on = "CAMPAIGN")
    campaign_data.drop("DESCRIPTION_y", axis = 1, inplace = True)
    campaign_data.rename(columns = {"DESCRIPTION_x" : "DESCRIPTION"}, inplace = True)
    campaign_data = campaign_data[campaign_data["COUPON_UPC"] > 0]
    category_data = pd.merge(category_data, campaign_data, how = "left", on = ["household_key", "PRODUCT_ID"])
    category_data["CAMPAIGN"] = np.where((category_data["START_DAY"] <= category_data["DAY"]) & 
                                           (category_data["DAY"] <= category_data["END_DAY"]), 
                                           category_data["CAMPAIGN"], np.NaN)
    category_data["DESCRIPTION"] = np.where((category_data["START_DAY"] <= category_data["DAY"]) & 
                                           (category_data["DAY"] <= category_data["END_DAY"]), 
                                           category_data["DESCRIPTION"], np.NaN)
    category_data.drop(labels = ["MANUFACTURER_y", "DEPARTMENT_y", "BRAND_y", "COMMODITY_DESC_y"], 
                       axis = 1, inplace = True)
    category_data.sort_values(by = ["household_key", "PRODUCT_ID", "BASKET_ID", "CAMPAIGN"], inplace = True)
    category_data.drop_duplicates(subset = ["household_key", "PRODUCT_ID", "BASKET_ID"], keep = "first", inplace = True)
    
    #dummify campaign and description of campaign
    dummies_campaign = pd.get_dummies(category_data["CAMPAIGN"], prefix = "campaign")
    dummies_description = pd.get_dummies(category_data["DESCRIPTION"], prefix = "description")
    category_data = pd.concat((category_data, dummies_campaign), axis = 1)
    category_data = pd.concat((category_data, dummies_description), axis = 1)
    
    category_data.drop(labels = ["STORE_ID", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC",
                             "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT", 
                             "COUPON_UPC", "CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"], 
                       axis = 1, inplace = True)
    category_data.rename(columns = {"MANUFACTURER_x" : "MANUFACTURER", "DEPARTMENT_x" : "DEPARTMENT", "BRAND_x" : "BRAND",
                               "COMMODITY_DESC_x" : "COMMODITY_DESC"},
                     inplace = True)
    
    return category_data


def sales_per_week(data):
    #calculate overall sales per week to determine if all weeks can be used (obviously, the first ~12 weeks sometimes look too low)
    weekly_data = data[["WEEK_NO", "SALES_VALUE"]].groupby("WEEK_NO").sum()
   
    return weekly_data.plot.bar(figsize=(20,10)), weekly_data.rolling(30).mean().plot(figsize=(20,10))


def remove_first_weeks(data, number):
    #remove first x weeks from data (see sales_per_week)
    data.drop(data[data.WEEK_NO <= number].index, inplace = True)
    
    return data   


def average_purchase_frequency(data):
    import numpy as np
    import pandas as pd
    
    #calculate average purchase frequency for category to determine how many weeks to cut off from the beginning of the data 
    purchases = pd.DataFrame(data.groupby(["household_key", "PRODUCT_ID"])["WEEK_NO"]
                             .agg(["min" , "max" , "count"]))
    
    for index, row in purchases.iterrows():
        if row["count"] > 1:
            purchases.loc[index, "time_lag"] = row["max"] - row["min"]
            purchases.loc[index, "freq_weeks"] = purchases.loc[index, "time_lag"] / row["count"]
        else:
            continue
    
    return purchases["freq_weeks"].mean(), purchases["freq_weeks"].hist()


def remove_weeks_first_purchase_cycle(data, number):
    #remove first x weeks from data (see sales_per_week); use 1.5 to 2 average purchase cycles
    data.drop(data[data.WEEK_NO <= number].index, inplace = True)
    
    return data   


def produce_dataset_w_ordinals(data):
    #remove samples without household demographics
    df = data[data.age_ordinal > 0]   
    #drop dummy variables for age, income and household size
    df.drop(labels = ["age_19-24", "age_25-34", "age_35-44", "age_45-54", "age_55-64", "age_65+",
                        "income_100-124K", "income_125-149K", "income_15-24K", "income_150-174K", "income_175-199K",
                        "income_200-249K", "income_25-34K", "income_250K+", "income_35-49K", "income_50-74K",
                        "income_75-99K", "income_Under 15K",
                        "hhsize_1", "hhsize_2", "hhsize_3", "hhsize_4", "hhsize_5+"],
              axis = 1, inplace = True)
    return df


def produce_dataset_w_dummies(data):  
    #remove samples without household demographics
    df = data[data.age_ordinal > 0]   
    #drop ordinal variables for age, income and household size
    df.drop(labels = ["age_ordinal", "income_ordinal", "hhsize_ordinal"],
              axis = 1, inplace = True)
    return df


def prepare_data_for_modelling(data):
    #drop all variables not used for modelling
    df = data.drop(labels = ["Unnamed: 0", "household_key", "SALES_VALUE", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "TRANS_TIME", "WEEK_NO",
                        "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY", "SUB_COMMODITY", "CURRENT_SIZE_OF_PRODUCT",
                        "paid_price", "COMMODITY_DESC"],
              axis = 1)
       
    return df