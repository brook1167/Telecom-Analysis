import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Image
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, normalize

def load_df(filepath):
    return pd.read_csv(filepath)

def get_missing_colum_with_percentage(df):
    num_missing = df.isnull().sum()
    num_rows = df.shape[0]

    data = {
        'num_missing': num_missing, 
        'percent_missing (%)': [round(x, 2) for x in num_missing / num_rows * 100]
    }

    stats = pd.DataFrame(data)

    # Filter columns with missing values
    return stats[stats['num_missing'] != 0]


def drop_nan(df):
    # Define the columns to check for missing values
    columns_to_check = ['Bearer Id', 'Start', 'End', 'IMSI', 'MSISDN/Number']
    
    # Drop rows where any of the specified columns have missing values, in place
    df.dropna(subset=columns_to_check, inplace=True)

def fill_missing_values(df):
    column_list =  [
        'DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)',
        '250 Kbps < DL TP < 1 Mbps (%)','DL TP > 1 Mbps (%)',
        'UL TP < 10 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)',
        '50 Kbps < UL TP < 300 Kbps (%)','UL TP > 300 Kbps (%)',
        'Last Location Name','Avg RTT DL (ms)','Avg RTT UL (ms)',
        'Nb of sec with Vol DL < 6250B','Nb of sec with Vol UL < 1250B'
    ]

    for column in column_list:
        if column != "Last Location Name":
            # Fill missing values with the mean for numeric columns, in place
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            # Fill missing values with the mode for categorical columns, in place
            df[column].fillna(df[column].mode()[0], inplace=True)

def fill_missing_values_t3(df):
    # Define columns to fill with their respective methods
    columns_to_fill = {
        'MSISDN/Number':'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Total_Avg_RTT': 'mean'
    }
    
    # Handle numeric columns
    for col, method in columns_to_fill.items():
        if col in df.columns:
            if method == 'mean':
                mean_value = df[col].mean()
                df[col] = df[col].fillna(value=mean_value)
            # Add more methods if needed
    
    # Handle categorical columns separately
    categorical_columns = ['Handset Type']
    
    for col in categorical_columns:
        if col in df.columns:
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col] = df[col].fillna(value=mode_value)


def calculate_total_data_volumes(df):
 
    # Calculate total data volumes for each application
    df['total_google'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)']
    df['total_email'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)']
    df['total_gaming'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']
    df['total_youtube'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
    df['total_netflix'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
    df['total_social'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
    df['total_other'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']

    # Aggregate total data volume per application by user
    total_data_volume_per_user = {
        'total_google': df.groupby('MSISDN/Number')['total_google'].sum(),
        'total_email': df.groupby('MSISDN/Number')['total_email'].sum(),
        'total_gaming': df.groupby('MSISDN/Number')['total_gaming'].sum(),
        'total_youtube': df.groupby('MSISDN/Number')['total_youtube'].sum(),
        'total_netflix': df.groupby('MSISDN/Number')['total_netflix'].sum(),
        'total_social': df.groupby('MSISDN/Number')['total_social'].sum(),
        'total_other': df.groupby('MSISDN/Number')['total_other'].sum()
    }

    # Print the aggregated data
    for app, data in total_data_volume_per_user.items():
        print(f'{app}:')
        print(data, end='\n\n')
    
    return total_data_volume_per_user


def top_five_decile(df):

    # Aggregate session data by user
    user_data = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()
    
    # Calculate total data (DL + UL)
    user_data['Total Data (Bytes)'] = user_data['Total DL (Bytes)'] + user_data['Total UL (Bytes)']

    # Sort users by total duration in descending order
    df_sorted = user_data.sort_values(by='Dur. (ms)', ascending=False)
    
    # Determine the top five deciles (deciles 6 to 10)
    df_sorted['Decile'] = pd.qcut(df_sorted['Dur. (ms)'], 10, labels=False) + 1
    
    # Filter for the top five deciles (6 to 10)
    df_top_five_deciles = df_sorted[df_sorted['Decile'] > 5]
    
    # Calculate total data (DL + UL) per decile class
    decile_data = df_top_five_deciles.groupby('Decile')['Total Data (Bytes)'].sum().reset_index()
    
    return decile_data


def calculate_statistics(df):
    # relevant columns are in the DataFrame
    relevant_features = ['Dur. (ms)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)', 'total_google', 'total_email', 'total_gaming', 'total_youtube', 'total_netflix', 'total_social', 'total_other']
    
    # Filter DataFrame to include only relevant columns
    df = df[relevant_features]
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Drop rows with NaN values for a cleaner calculation
    numeric_df = numeric_df.dropna()

    # Calculate and print mean for numeric columns
    print("Mean:\n", numeric_df.mean(), '\n')
    
    # Calculate and print median for numeric columns
    print("Median:\n", numeric_df.median(), '\n')
    
    # Calculate and print mode for numeric columns
    mode_df = numeric_df.mode()
    if not mode_df.empty:
        mode_series = mode_df.iloc[0]
        print("Mode:\n", mode_series, '\n')
    else:
        print("Mode: No mode found\n")
    
    # Calculate and print highest values for numeric columns
    print("Highest:\n", numeric_df.max(), '\n')
    
    # Calculate and print lowest values for numeric columns
    print("Lowest:\n", numeric_df.min(), '\n')


def non_graphical_analysis(df):
    numeric_df = df.select_dtypes(include=['number'])
    
    # Get the summary statistics
    desc = numeric_df.describe()
    
    # Calculate additional statistics
    desc.loc['range'] = desc.loc['max'] - desc.loc['min']
    desc.loc['variance'] = numeric_df.var()
    desc.loc['IQR'] = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    
    return desc


def calculate_total_data(df):
   
    # Calculate total data for each category
    df["Youtube_Total_Data"] = df["Youtube DL (Bytes)"] + df["Youtube UL (Bytes)"]
    df["Google_Total_Data"] = df["Google DL (Bytes)"] + df["Google UL (Bytes)"]
    df["Email_Total_Data"] = df["Email DL (Bytes)"] + df["Email UL (Bytes)"]
    df["Social_Media_Total_Data"] = df["Social Media DL (Bytes)"] + df["Social Media UL (Bytes)"]
    df["Netflix_Total_Data"] = df["Netflix DL (Bytes)"] + df["Netflix UL (Bytes)"]
    df["Gaming_Total_Data"] = df["Gaming DL (Bytes)"] + df["Gaming UL (Bytes)"]
    df["Other_Total_Data"] = df["Other DL (Bytes)"] + df["Other UL (Bytes)"]
    df["Total_UL_and_DL"] = df["Total UL (Bytes)"] + df["Total DL (Bytes)"]
    
    return df


def plot_scatter(df, x_col, y_col, title) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_heatmap(df,title, cbar=False):
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_hist(df,column, color):
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def mult_hist(sr, rows, cols, title_text, subplot_titles, interactive=False):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    for i in range(rows):
        for j in range(cols):
            x = ["-> " + str(i) for i in sr[i+j].index]
            fig.add_trace(go.Bar(x=x, y=sr[i+j].values), row=i+1, col=j+1)
    fig.update_layout(showlegend=False, title_text=title_text)
    if(interactive):
        fig.show()
    else:
        return Image(pio.to_image(fig, format='png', width=1200))


def plot_top_applications():
    # Define data
    applications = ['Gaming', 'Youtube', 'Netflix', 'Google', 'Email', 'Social Media', 'Other']
    data_values = [63910124731666, 3362537620065, 3360563670772, 1159544186717, 335503000148, 271478798103, 63770726326023]
    top_applications, top_values = zip(*sorted(zip(applications, data_values), key=lambda x: x[1], reverse=True)[:3])
    plt.figure(figsize=(8, 6))
    plt.bar(top_applications, top_values, color=['blue', 'green', 'orange'])
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('Applications')
    plt.ylabel('Total Data (in bytes)')
    for i, value in enumerate(top_values):
        plt.text(i, value + 0.002 * max(top_values), f'{value:,}', ha='center')
    plt.show()




def choose_kmeans(df: pd.DataFrame, num: int):
  distortions = []
  inertias = []
  K = range(1, num)
  for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    distortions.append(sum(
        np.min(cdist(df, kmeans.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    inertias.append(kmeans.inertia_)

  return (distortions, inertias)


def aggregate_customer_data(df):
    # Calculate average TCP retransmission
    df['Avg TCP Retrans (Bytes)'] = (df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']) / 2

    # Calculate average RTT
    df['Avg RTT (ms)'] = (df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']) / 2

    # Calculate average throughput
    df['Avg Throughput (kbps)'] = (df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']) / 2

    # Aggregate data per customer
    aggregated_df = df.groupby('MSISDN/Number').agg({
        'Avg TCP Retrans (Bytes)': 'mean',
        'Avg RTT (ms)': 'mean',
        'Handset Type': 'first',  # Take the first handset type for each customer
        'Avg Throughput (kbps)': 'mean'
    }).reset_index()

    return aggregated_df
def process_data(df):
    df['Total_Avg_RTT'] = df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']
    df['Total_Avg_Bearer_TP'] = df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']
    df['Total_Avg_TCP'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
    
    df['Total_Avg_RTT'].fillna(value=df['Total_Avg_RTT'].mean(), inplace=True)
    
    return df


def analyze_tcp(df):
    sorted_by_tcp = df.sort_values('Total_Avg_TCP', ascending=False)
    top_10 = sorted_by_tcp.head(10)['Total_Avg_TCP']
    last_10 = sorted_by_tcp.tail(10)['Total_Avg_TCP']
    most_10 = df['Total_Avg_TCP'].value_counts().head(10)
    
    return top_10, last_10, most_10

def analyze_rtt(df):
    sorted_by_RTT = df.sort_values('Total_Avg_RTT', ascending=False)
    top_10 = sorted_by_RTT.head(10)['Total_Avg_RTT']
    last_10 = sorted_by_RTT.tail(10)['Total_Avg_RTT']
    most_10 = df['Total_Avg_RTT'].value_counts().head(10)
    
    return top_10, last_10, most_10

def analyze_tp(df):
    sorted_by_Bearer_TP = df.sort_values('Total_Avg_Bearer_TP', ascending=False)
    top_10 = sorted_by_Bearer_TP.head(10)['Total_Avg_Bearer_TP']
    last_10 = sorted_by_Bearer_TP.tail(10)['Total_Avg_Bearer_TP']
    most_10 = df['Total_Avg_Bearer_TP'].value_counts().head(10)
    
    return top_10, last_10, most_10

def mult_hist(sr, rows, cols, title_text, subplot_titles, interactive=False):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    for i in range(rows):
        for j in range(cols):
            x = ["-> " + str(i) for i in sr[i+j].index]
            fig.add_trace(go.Bar(x=x, y=sr[i+j].values), row=i+1, col=j+1)
    fig.update_layout(showlegend=False, title_text=title_text)
    if(interactive):
        fig.show()
    else:
        return Image(pio.to_image(fig, format='png', width=1200))
    

def aggregate_handset_type(df):
    handset_type_agg = df.groupby('Handset Type').agg(
        {'Total_Avg_Bearer_TP': 'mean', 'Total_Avg_TCP': 'mean'})
    return handset_type_agg

