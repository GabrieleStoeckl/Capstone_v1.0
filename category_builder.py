import numpy as np
import pandas as pd

category = "CANDY - PACKAGED"

transactions = pd.read_csv("data/transaction_data.csv")
products = pd.read_csv("data/product.csv")

transactions = pd.merge(transactions, products, how = "left", on = "PRODUCT_ID")

condition = transactions["COMMODITY_DESC"] == category
cat_df = transactions[condition]

# each row is a purchase event. we will find the first purchase event of each product by household 
# use WEEK_NO, TRANS_TIME, household_key and PRODUCT_ID to find the first purchase event of a product by household
# use the cat_df dataframe:
(cat_df
 .sort_values(by=["household_key", "PRODUCT_ID", "WEEK_NO", "DAY", "TRANS_TIME"], 
              inplace=True)
)

# group the data by household_key and PRODUCT_ID
grouped = cat_df.groupby(["household_key", "PRODUCT_ID"])
# get the first purchase event of each product by household
cols_to_keep = ["household_key", "PRODUCT_ID", "WEEK_NO", "DAY", "TRANS_TIME", "BASKET_ID"]
first_purchase = grouped.first().reset_index()[cols_to_keep]

cat_df2 = (pd
                .merge(cat_df, first_purchase, 
                       on=["household_key", "PRODUCT_ID", "WEEK_NO", "DAY", "TRANS_TIME"], 
                       how="left")
               )

# create "first_purchase" column, True if there is no NaN in any of the columns, False otherwise
cat_df2["first_purchase"] = ~cat_df2.isnull().any(axis=1)

#print(cat_df2["first_purchase"].value_counts())

#calculate regular shelf price w/o discounts
cat_df2.loc[:, "shelf_price"] = (cat_df2.loc[:, "SALES_VALUE"] - cat_df2.loc[:, "RETAIL_DISC"] - cat_df2.loc[:, "COUPON_MATCH_DISC"]) / cat_df2.loc[:, "QUANTITY"]
#calculate paid shelf price with discounts
cat_df2.loc[:, "paid_price"] = (cat_df2.loc[:, "SALES_VALUE"] + cat_df2.loc[:, "COUPON_DISC"]) / cat_df2.loc[:, "QUANTITY"]
#calculate overall discount
cat_df2.loc[:, "pct_disc"] = (cat_df2.loc[:, "shelf_price"]-cat_df2.loc[:, "paid_price"]) / cat_df2.loc[:, "shelf_price"]
#calculate discount in percentage points versus regualar shelf price due to retailer loyalty card 
cat_df2.loc[:, "pct_retail_disc"] = (-cat_df2.loc[:, "RETAIL_DISC"] / cat_df2.loc[:, "QUANTITY"]) / cat_df2.loc[:, "shelf_price"]
#calculate discount in percentage points versus regualar shelf price due to manufacturer coupons 
cat_df2.loc[:, "pct_coupon_disc"] = ((-cat_df2.loc[:, "COUPON_DISC"]-cat_df2.loc[:, "COUPON_MATCH_DISC"]) / cat_df2.loc[:, "QUANTITY"] ) / cat_df2.loc[:, "shelf_price"]

# remove rows with negative paid price
cat_df2 = cat_df2[cat_df2["paid_price"] > 0]

#adding in causal data
causal = pd.read_csv("data/causal_data.csv")
causal_products = pd.merge(causal, products[["PRODUCT_ID", "COMMODITY_DESC"]], how = "left", on = "PRODUCT_ID")
causal_products.drop(causal_products[causal_products.COMMODITY_DESC != category].index, inplace = True)
cat_df3 = pd.merge(cat_df2, causal_products, how = "left", on = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
cat_df3.drop_duplicates(subset= ["household_key", "BASKET_ID_x", "STORE_ID", "DAY", "PRODUCT_ID"], keep = "last", inplace= True)


#dummy variables for display
dummies_display = pd.get_dummies(cat_df3["display"], prefix = "display")
cat_df3 = pd.concat((cat_df3, dummies_display), axis = 1)
#dummy variables for mailer
dummies_mailer = pd.get_dummies(cat_df3["mailer"], prefix = "mailer")
cat_df3 = pd.concat((cat_df3, dummies_mailer), axis = 1)
cat_df3.drop("COMMODITY_DESC_y", axis = 1, inplace = True)
cat_df3.rename(columns = {"COMMODITY_DESC_x" : "COMMODITY_DESC"}, inplace=True)

hhdemogs = pd.read_csv("data/hh_demographic.csv")
cat_df3 = pd.merge(cat_df3, hhdemogs, how = "left", on="household_key")
coupon_redempt = pd.read_csv("data/coupon_redempt.csv")
coupon = pd.read_csv("data/coupon.csv")
couponr_prodid = pd.merge(coupon_redempt, coupon, how = "left", on = ["COUPON_UPC" , "CAMPAIGN"])
cat_df3 = pd.merge(cat_df3, couponr_prodid, how = "left", on = ["household_key", "PRODUCT_ID", "DAY"])


dummies_marital_status = pd.get_dummies(cat_df3["MARITAL_STATUS_CODE"], prefix = "marital_status")
dummies_homeowner = pd.get_dummies(cat_df3["HOMEOWNER_DESC"], prefix = "homeowner")
dummies_hhcomp = pd.get_dummies(cat_df3["HH_COMP_DESC"], prefix = "hhcomp")
dummies_kidcat = pd.get_dummies(cat_df3["KID_CATEGORY_DESC"], prefix = "kid_category")
cat_df3 = pd.concat((cat_df3, dummies_marital_status), axis = 1)
cat_df3 = pd.concat((cat_df3, dummies_homeowner), axis = 1)
cat_df3 = pd.concat((cat_df3, dummies_hhcomp), axis = 1)
cat_df3 = pd.concat((cat_df3, dummies_kidcat), axis = 1)

map_age = {"19-24":1, "25-34":2, "35-44":3, "45-54":4, "55-64":5, "65+":6}
cat_df3 = cat_df3.assign(age_ordinal = cat_df3.AGE_DESC.map(map_age))

map_income = {"Under 15K":10, "15-24K":19.5, "25-34K":29.5, "35-49K":39.5, "50-74K":62, "75-99K":87, "100-124K":112, 
              "125-149K":137, "150-174K":162, "175-199K":187, "200-249K":224.5, "250K+":250}
cat_df3 = cat_df3.assign(income_ordinal = cat_df3.INCOME_DESC.map(map_income))

map_hhsize = {"1":1, "2":2, "3":3, "4":4, "5+":5}
cat_df3 = cat_df3.assign(hhsize_ordinal = cat_df3.HOUSEHOLD_SIZE_DESC.map(map_hhsize))


# save the data to a csv file
path = f"data/{category}_transactions.csv"
cat_df3.to_csv(path, index=False)
