import pandas as pd
import streamlit as st
import plotly.express as px
import os
import sys

# Add the path to the directory
rpath = os.path.abspath('..')
if rpath not in sys.path: 
    sys.path.insert(0, rpath)

# Load CSV
df = pd.read_csv("./data/telecom.csv")

# Function to drop rows with missing values in specific columns
def drop_nan(df):
    columns_to_check = ['Bearer Id', 'Start', 'End', 'IMSI', 'MSISDN/Number']
    df.dropna(subset=columns_to_check, inplace=True)

# Function to fill missing values
def fill_missing_values(df):
    column_list = [
        'DL TP < 50 Kbps (%)','50 Kbps < DL TP < 250 Kbps (%)',
        '250 Kbps < DL TP < 1 Mbps (%)','DL TP > 1 Mbps (%)',
        'UL TP < 10 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)',
        '50 Kbps < UL TP < 300 Kbps (%)','UL TP > 300 Kbps (%)',
        'Last Location Name','Avg RTT DL (ms)','Avg RTT UL (ms)',
        'Nb of sec with Vol DL < 6250B','Nb of sec with Vol UL < 1250B'
    ]

    for column in column_list:
        if column != "Last Location Name":
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

def fill_missing_values_t3(df):
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
    
    for col, method in columns_to_fill.items():
        if col in df.columns:
            if method == 'mean':
                mean_value = df[col].mean()
                df[col] = df[col].fillna(value=mean_value)
    
    categorical_columns = ['Handset Type']
    
    for col in categorical_columns:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(value=mode_value)

# Set the title of the Streamlit app
st.title("Analysis Dashboard")

# Add a selectbox in the sidebar for analysis options
st.sidebar.title("Navigation")
analysis_options = ["Explore Dataset", "User Overview Analysis", "User Engagement Analysis", "User Experience Analysis"]
selected_analysis = st.sidebar.selectbox("Choose an analysis:", analysis_options)



# Explore Dataset the data
if selected_analysis == "Explore Dataset":
    st.write("You are viewing raw data.")
    
    # Add a dropdown menu for dataset exploration options
    exploration_options = ["Head", "Tail", "Describe", "Count of Missing Values"]
    selected_option = st.selectbox("Choose an option to explore the dataset:", exploration_options)
    
    if selected_option == "Head":
        st.write(df.head())
    elif selected_option == "Tail":
        st.write(df.tail())
    elif selected_option == "Describe":
        st.write(df.describe())
    elif selected_option == "Count of Missing Values":
        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("Count of Missing Values:")
            missing_values = df.isnull().sum()
            st.write(missing_values)

            handle_missing_options = [
                "Do Nothing", 
                "Drop Rows with Missing Values",
                "Fill with Mean/Mode",
                "Fill with Specific Methods"
            ]
            selected_missing_option = st.selectbox("Choose how to handle missing values:", handle_missing_options)
            
            if selected_missing_option == "Drop Rows with Missing Values":
                drop_nan(df)
                st.write("Rows with missing values in the specified columns have been dropped.")
                st.write(df.isnull().sum())

            elif selected_missing_option == "Fill with Mean/Mode":
                fill_missing_values(df)
                st.write("Missing values have been filled with mean (for numeric) or mode (for categorical).")
                st.write(df.isnull().sum())

            elif selected_missing_option == "Fill with Specific Methods":
                fill_missing_values_t3(df)
                st.write("Missing values have been filled using specific methods.")
                st.write(df.isnull().sum())

        with col2:
            st.write("Bar Chart of Missing Values:")
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                st.bar_chart(missing_values)
            else:
                st.write("No missing values found.")

# Display User Overview Analysis
elif selected_analysis == "User Overview Analysis":

    

    st.write("Top 10 handsets used by customers:")
    
    # Calculate top 10 handsets and sort them
    handset_type = df['Handset Type'].value_counts().head(10).sort_values(ascending=True)
    
    # Create columns to control layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display the bar chart using Streamlit's native chart, larger column
        st.bar_chart(handset_type)

    with col2:
        # Display the top 10 handsets
        st.write(handset_type)

    st.write("Top 3 handset manufacturers:")
    
    # Calculate top 3 handset manufacturers and sort them
    handset_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    
    # Create columns to control layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display the bar chart using Streamlit's native chart, larger column
        st.bar_chart(handset_manufacturers)

    with col2:
        # Display the top 3 handset manufacturers
        st.write(handset_manufacturers)

    st.write("Top 5 handsets per top 3 handset manufacturers:")

    # Top 5 handsets for Apple
    top_apple = df[df['Handset Manufacturer'] == 'Apple']
    top_apple = top_apple.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
    top_apple = top_apple.nlargest(5, 'count')
    
    st.write("Top 5 Apple Handsets:")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(top_apple.set_index('Handset Type')['count'])
    with col2:
        st.write(top_apple)
    
    # Top 5 handsets for Samsung
    top_samsung = df[df['Handset Manufacturer'] == 'Samsung']
    top_samsung = top_samsung.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
    top_samsung = top_samsung.nlargest(5, 'count')
    
    st.write("Top 5 Samsung Handsets:")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(top_samsung.set_index('Handset Type')['count'])
    with col2:
        st.write(top_samsung)
    
    # Top 5 handsets for Huawei
    top_huawei = df[df['Handset Manufacturer'] == 'Huawei']
    top_huawei = top_huawei.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
    top_huawei = top_huawei.nlargest(5, 'count')
    
    st.write("Top 5 Huawei Handsets:")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(top_huawei.set_index('Handset Type')['count'])
    with col2:
        st.write(top_huawei)

    st.write("Histogram of Total UL (Bytes)")


    # Histogram of 'Total UL (Bytes)' using Plotly
    fig_ul_bytes = px.histogram(df, x='Total UL (Bytes)', nbins=30, title="Distribution of Total UL (Bytes)")
    st.plotly_chart(fig_ul_bytes)

    st.write("Histogram of Total DL (Bytes)")
    
    # Histogram of 'Total DL (Bytes)' using Plotly
    fig_dl_bytes = px.histogram(df, x='Total DL (Bytes)', nbins=30, title="Distribution of Total DL (Bytes)")
    st.plotly_chart(fig_dl_bytes)

    st.write("Histogram of Social_Media_Total_Data")

    df["Social_Media_Total_Data"] = df["Social Media DL (Bytes)"] + df["Social Media UL (Bytes)"]

    # Histogram of 'Social_Media_Total_Data' using Plotly
    fig_social_media = px.histogram(df, x='Social_Media_Total_Data', nbins=30, title="Distribution of Social Media Total Data")
    st.plotly_chart(fig_social_media)

    
    st.write("Histogram of Google_Total_Data")

    df["Google_Total_Data"] = df["Google DL (Bytes)"] + df["Google UL (Bytes)"]
    
    # Histogram of 'Google_Total_Data' using Plotly
    fig_google_data = px.histogram(df, x='Google_Total_Data', nbins=30, title="Distribution of Google Total Data")
    st.plotly_chart(fig_google_data)

    
    st.write("Histogram of Email_Total_Data")

    df["Email_Total_Data"] = df["Email DL (Bytes)"] + df["Email UL (Bytes)"]
    
    # Histogram of 'Email_Total_Data' using Plotly
    fig_email_data = px.histogram(df, x='Email_Total_Data', nbins=30, title="Distribution of Email Total Data")
    st.plotly_chart(fig_email_data)

    st.write("Histogram of Youtube_Total_Data")

    df["Youtube_Total_Data"] = df["Youtube DL (Bytes)"] + df["Youtube UL (Bytes)"]
    
    # Histogram of 'Email_Total_Data' using Plotly
    fig_youtube_data = px.histogram(df, x='Youtube_Total_Data', nbins=30, title="Distribution of Youtube Total Data")
    st.plotly_chart(fig_youtube_data)



    st.write("Histogram of Total Netflix Data")

    df["Netflix_Total_Data"] = df["Netflix DL (Bytes)"] + df["Netflix UL (Bytes)"]
    
    # Histogram of 'Netflix_Total_Data' using Plotly
    fig_netflix_data = px.histogram(df, x='Netflix_Total_Data', nbins=30, title="Distribution of Netflix Total Data")
    st.plotly_chart(fig_netflix_data)

    st.write("Histogram of Total Gaming Data")

    df["Gaming_Total_Data"] = df["Gaming DL (Bytes)"] + df["Gaming UL (Bytes)"]
    
    # Histogram of 'Gaming_Total_Data' using Plotly
    fig_gaming_data = px.histogram(df, x='Gaming_Total_Data', nbins=30, title="Distribution of Gaming Total Data")
    st.plotly_chart(fig_gaming_data)

    st.write("Histogram of Other Data")

    df["Other_Total_Data"] = df["Other DL (Bytes)"] + df["Other UL (Bytes)"]
    
    # Histogram of 'Other_Total_Data' using Plotly
    fig_other_data = px.histogram(df, x='Other_Total_Data', nbins=30, title="Distribution of Other Total Data")
    st.plotly_chart(fig_other_data)

    st.write("Scatter Plot of Total UL (Bytes) vs. Gaming_Total_Data")

    df["Total_UL_and_DL"] = df["Total UL (Bytes)"] + df["Total DL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Gaming_Total_Data', title="Total UL (Bytes) vs. Gaming_Total_Data",)
    st.plotly_chart(fig_scatter)

    st.write("Scatter Plot of Total UL (Bytes) vs. Youtube Data")

    df["Youtube_Total_Data"] = df["Youtube DL (Bytes)"] + df["Youtube UL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Youtube_Total_Data', title="Total UL (Bytes) vs. Youtube_Total_Data",)
    st.plotly_chart(fig_scatter)

    st.write("Scatter Plot of Total Data vs. Email Total Data")

    df["Email_Total_Data"] = df["Email DL (Bytes)"] + df["Email UL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Email_Total_Data', title="Total Data vs. Email_Total_Data",)
    st.plotly_chart(fig_scatter)

    st.write("Scatter Plot of Total Data vs. Social Media Total Data")

    df["Social_Media_Total_Data"] = df["Social Media DL (Bytes)"] + df["Social Media UL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Social_Media_Total_Data', title="Total Data vs. Social_Media_Total_Data",)
    st.plotly_chart(fig_scatter)

    st.write("Scatter Plot of Total Data Vs. Netflix_Total_Data (MegaBytes)")

    df["Netflix_Total_Data"] = df["Netflix DL (Bytes)"] + df["Netflix UL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Netflix_Total_Data', title="Total Data Vs. Netflix_Total_Data (MegaBytes)",)
    st.plotly_chart(fig_scatter)

    st.write("Scatter Plot of Total Data Vs. Other_Total_Data")

    df["Other_Total_Data"] = df["Other DL (Bytes)"] + df["Other UL (Bytes)"]

    # Scatter plot of 'Total UL (Bytes)' vs 'Total DL (Bytes)' using Plotly
    fig_scatter = px.scatter(df, x='Total_UL_and_DL', y='Other_Total_Data', title="Total Data Vs. Other_Total_Data",)
    st.plotly_chart(fig_scatter)

    st.write("Correlation of Usage of User Data Volume")
    columns = ['Youtube_Total_Data', 'Google_Total_Data', 'Email_Total_Data','Social_Media_Total_Data', 'Netflix_Total_Data', 'Gaming_Total_Data', 'Other_Total_Data', 'Total_UL_and_DL']

    selected_columns_df = df[columns]

    # Generate the correlation matrix for the specified columns
    corr_matrix = selected_columns_df.corr()

# Generate heatmap using Plotly
    fig = px.imshow(corr_matrix, 
                text_auto=True, 
                color_continuous_scale='Viridis', 
                labels={'color': 'Correlation'},
                title='Heatmap of Correlation Between Columns')

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)
    

# Placeholder for content based on selected analysis
elif selected_analysis == "User Engagement Analysis":
    st.write("User Engagement Analysis details")

    df["Total_UL_and_DL"] = df["Total UL (Bytes)"] + df["Total DL (Bytes)"]
    # Prepare user data
    users = df[['MSISDN/Number', 'Bearer Id', 'Dur. (ms)', 'Total_UL_and_DL']].copy().rename(columns={'Dur. (ms)': 'time_duration'})
    users = users.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',
        'time_duration': 'sum',
        'Total_UL_and_DL': 'sum'
    }).rename(columns={'Bearer Id': 'sessions'})

    # Display top 10 users
    top_10_users = users.head(10)

    # Bar chart for total traffic (upload + download) per user
    st.write("Top 10 Users by Total Upload and Download:")
    st.bar_chart(top_10_users[['Total_UL_and_DL']])

    # Bar chart for total sessions
    st.write("Top 10 Users by Total Sessions:")
    st.bar_chart(top_10_users[['sessions']])

    # Bar chart for time duration per user
    st.write("Top 10 Users by Time Duration:")
    st.bar_chart(top_10_users[['time_duration']])

    # Perpare the data frames
    df["Youtube_Total_Data"] = df["Youtube DL (Bytes)"] + df["Youtube UL (Bytes)"]
    df["Google_Total_Data"] = df["Google DL (Bytes)"] + df["Google UL (Bytes)"]
    df["Email_Total_Data"] = df["Email DL (Bytes)"] + df["Email UL (Bytes)"]
    df["Social_Media_Total_Data"] = df["Social Media DL (Bytes)"] + df["Social Media UL (Bytes)"]
    df["Netflix_Total_Data"] = df["Netflix DL (Bytes)"] + df["Netflix UL (Bytes)"]
    df["Gaming_Total_Data"] = df["Gaming DL (Bytes)"] + df["Gaming UL (Bytes)"]
    df["Other_Total_Data"] = df["Other DL (Bytes)"] + df["Other UL (Bytes)"]    

     # Prepare application usage data
    apps = df.groupby('MSISDN/Number').agg({
        'Gaming_Total_Data': 'sum',
        'Youtube_Total_Data': 'sum',
        'Netflix_Total_Data': 'sum',
        'Google_Total_Data': 'sum',
        'Email_Total_Data': 'sum',
        'Social_Media_Total_Data': 'sum',
        'Other_Total_Data': 'sum'
    })

    # Calculate the total data usage per application
    apps_total = apps.sum().sort_values(ascending=False)
    
    # Get top 3 most used applications
    top_3_apps = apps_total.head(3)
    

    # Bar chart for top 3 most used applications
    st.write("Top 3 Most Used Applications:")
    st.bar_chart(top_3_apps)

elif selected_analysis == "User Experience Analysis":
    
    # You can add more content specific to User Experience Analysis here
    st.write("You are viewing User Experience Analysis data.")
    
    # Fix missing values
    fill_missing_values_t3(df)
    
    # Function to analyze TCP,RTT,TP data
    def analyze_tcp(df):
        df['Total_Avg_TCP'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
        
        sorted_by_tcp = df.sort_values('Total_Avg_TCP', ascending=False)
        top_10_tcp = sorted_by_tcp.head(10)['Total_Avg_TCP']
        last_10_tcp = sorted_by_tcp.tail(10)['Total_Avg_TCP']
        most_10_tcp = df['Total_Avg_TCP'].value_counts().head(10)
    
        return top_10_tcp, last_10_tcp, most_10_tcp
    def analyze_rtt(df):
        df['Total_Avg_RTT'] = df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']
        sorted_by_RTT = df.sort_values('Total_Avg_RTT', ascending=False)
        top_10_rtt = sorted_by_RTT.head(10)['Total_Avg_RTT']
        last_10_rtt = sorted_by_RTT.tail(10)['Total_Avg_RTT']
        most_10_rtt = df['Total_Avg_RTT'].value_counts().head(10)
        return top_10_rtt, last_10_rtt, most_10_rtt

    def analyze_tp(df):
        df['Total_Avg_Bearer_TP'] = df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']
        sorted_by_Bearer_TP = df.sort_values('Total_Avg_Bearer_TP', ascending=False)
        top_10_tp = sorted_by_Bearer_TP.head(10)['Total_Avg_Bearer_TP']
        last_10_tp = sorted_by_Bearer_TP.tail(10)['Total_Avg_Bearer_TP']
        most_10_tp = df['Total_Avg_Bearer_TP'].value_counts().head(10)
    
        return top_10_tp, last_10_tp, most_10_tp
    
    
    # Analyze TCP values
    top_10_tcp, last_10_tcp, most_10_tcp = analyze_tcp(df)

    # Plot TCP values
    st.write("Top 10 TCP  Values:")
    st.bar_chart(top_10_tcp)

     # Analyze RTT values
    top_10_rtt, last_10_rtt, most_10_rtt = analyze_rtt(df)
    
    # Plot RTT values
    st.write("Top 10 RTT Retransmission Values:")
    st.bar_chart(top_10_rtt)

    # Analyze TP values
    top_10_tp, last_10_tp, most_10_tp = analyze_tp(df)

    # Plot TP values
    st.write("Top 10 Throughput Values:")
    st.bar_chart(top_10_tp)

     # Plot TCP values
    st.write("Last 10 TCP  Values:")
    st.bar_chart(last_10_tcp)

    # Plot RTT values
    st.write("Last 10 RTT Retransmission Values:")
    st.bar_chart(last_10_rtt)

    # Plot TP values
    st.write("Last 10 TP  Values:")
    st.bar_chart(last_10_tp)


    # Plot TCP values
    st.write("Most 10 TCP  Values:")
    st.bar_chart(most_10_tcp)

    # Plot RTT values
    st.write("Most 10 RTT Retransmission Values:")
    st.bar_chart(most_10_rtt)

    # Plot TP values
    st.write("Most 10 TP  Values:")
    st.bar_chart(most_10_tp)

   