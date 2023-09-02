import gradio as gr
import pickle
import pandas as pd
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use("ggplot")
import sklearn 
from sklearn.decomposition import TruncatedSVD

# Load the model from the .pkl file
with open('collaborative_filtering_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data from the CSV file
df = pd.read_csv('BigBasket Products.csv')

df.drop(columns=['sub_category','brand','type','description','sale_price','market_price'],inplace=True)

df.dropna(inplace=True)

df.sort_values(["product"], 
                    axis=0,
                    ascending=[False], 
                    inplace=True)

product=list(df['product'])

df['product'].value_counts()

count=1
productid=[]

productid.append(1)

for i in range(1,len(product)):
    if product[i]==product[i-1]: 
        productid.append(count)
    else: 
        count=count+1
        productid.append(count)

df['product_id']=productid

pop_products = pd.DataFrame(df.groupby('product_id')['rating'].count())
most_pop = pop_products.sort_values('rating', ascending=False)

subdf=df.head(1000000)

ratings_utility_matrix = subdf.pivot_table(values='rating',index='index',columns='product_id',fill_value=0)

rating_utility_matrix_transpose = ratings_utility_matrix.T

X=rating_utility_matrix_transpose

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

correlation_matrix = np.corrcoef(decomposed_matrix)

size=correlation_matrix.shape[0]


# Define a function to make recommendations based on a user's input
def model(product_id): 
    correlation_product_id = correlation_matrix[product_id]
    recommend = list(X.index[correlation_product_id > 0.80])
    rlist=[]
    for item in recommend: 
        rlist.append(df.loc[df['product_id']==item,'product'])
    return rlist


# Define the input and output interfaces for Gradio
input_interface = gr.inputs.Number(default=1, label="User ID")
output_interface = gr.outputs.Label(num_top_classes=5, label="Top 5 Recommendations")

# Create the Gradio app
gradio_app = gr.Interface(model, inputs=input_interface, outputs=output_interface, title="Recommendation System", 
                          description="Enter a Product ID to receive personalized recommendations.")

# Launch the app
gradio_app.launch()








