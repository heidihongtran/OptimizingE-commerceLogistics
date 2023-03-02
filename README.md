# Project: Optimizing E-commerce Logistics
This is a group work in my post-graduate program.

The goal of this assignment is to collect supply chain-related data and build models to help a company better practice understanding, efficiency, and effectiveness. That can be achieved by enabling data-driven decisions at strategic, operational, and tactical levels. 

Analytics can directly contribute to the Supply Chain processes. We chose this dataset to analyze and optimize the shipment status. 

Data source: https://www.kaggle.com/datasets/prachi13/customer-analytics

A Look Into The Dataset:

This dataset is accessible in Kaggle and includes 10,999 observations across 12 columns.
It was collected by a worldwide electronics e-commerce company, who aims to study their
customers and discover key insights from the database using some of the most advanced machine
learning methodologies. Product shipment tracking is the focus of the data.

Every observation displays a unique customer ID with the corresponding warehouse
block A, B, C, D, and E, where their goods are stored, the product weight in grams, the cost of
the shipped product in U.S. dollars, the mode of transport (ship, flight, or road), and the discount
offered on that specific product. On top of that, Reached.on.Time_Y.N is detected as the target
variable, which tells whether the product has arrived on schedule (0) or not (1). If we look at the
Customer_rating column, we can see how the firm values its clients on a scale of 1 (the worst) to
5 (the best). Using the Product_importance attribute, the company also assigns a low, medium, or
high category to the shipped product. The other variables are Gender, Prior_purchases showing
the number of prior purchases, and Customer_care_call depicting the number of calls made for
shipment inquiries.

In terms of the data type, all 12 original variables in the dataset are classified into two
main categories: categorical variables like Warehouse_block, Mode_of_Shipment,
Product_importance, and Gender and numerical variables like ID, Customer_care_calls,
Customer_rating, Cost_of_the_Product, Prior_purchases, Discount_offered, Weight_in_gms, and
Reached.on.Time_Y.N.
