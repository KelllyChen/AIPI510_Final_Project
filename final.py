import unittest
from statsmodels.stats.power import TTestIndPower
from googleapiclient.discovery import build
import pandas as pd
import isodate
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
from scipy.spatial import distance
from unittest.mock import patch, MagicMock
import os

# Define the YouTube API client as a class
class YouTubeDataAnalyzer:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    # Power analysis
    def power_analysis(self):
        # Set parameters
        effect_size = 0.5  # Medium effect size for correlation
        alpha = 0.05  # Significance level
        power = 0.8  # Desired power

        # Perform power analysis
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)

        print(f"Required sample size: {np.ceil(sample_size)}")

    # Search channel id based on channel name
    def search_youtuber_channel_id(self, name):
        request = self.youtube.search().list(part='snippet', q=name, type='channel', maxResults=1)
        response = request.execute()
        return response['items'][0]['snippet']['channelId']

    # Get video ids from specific channel
    def get_videos(self, channel_id, max_results):
        channel_request = self.youtube.channels().list(part='contentDetails', id=channel_id)
        channel_response = channel_request.execute()

        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        video_ids = []
        next_page_token = None

        while len(video_ids) < max_results:
            playlist_request = self.youtube.playlistItems().list(
                part='snippet', 
                playlistId=uploads_playlist_id, 
                maxResults=100, 
                pageToken=next_page_token
            )
            playlist_response = playlist_request.execute()

            video_ids.extend(item['snippet']['resourceId']['videoId'] for item in playlist_response['items'])

            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break

        return video_ids[:max_results]

    # Get video details and filter out shorts
    def get_video_details(self, video_ids):
        videos_data = []
        for video_id in video_ids:
            video_request = self.youtube.videos().list(part='snippet,statistics,contentDetails', id=video_id)
            video_response = video_request.execute()

            for video in video_response['items']:
                duration = video['contentDetails']['duration']
                # Convert duration to seconds
                duration_seconds = isodate.parse_duration(duration).total_seconds()

                # Filter out shorts(<=60sec)
                if duration_seconds <= 60:
                    continue

                video_info = {
                    'title': video['snippet']['title'],
                    'publish_date': video['snippet']['publishedAt'],
                    'tags': ','.join(video['snippet'].get('tags', [])),
                    'thumbnail_url': video['snippet']['thumbnails']['high']['url'],
                    'view_count': video['statistics'].get('viewCount', 0),
                    'like_count': video['statistics'].get('likeCount', 0),
                    'comment_count': video['statistics'].get('commentCount', 0),
                }
                videos_data.append(video_info)
        return videos_data

    # Save video data to csv
    def save_to_csv(self, videos_data, filename):
        df = pd.DataFrame(videos_data)
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['view_count'] = pd.to_numeric(df['view_count'])
        df['like_count'] = pd.to_numeric(df['like_count'])
        df['comment_count'] = pd.to_numeric(df['comment_count'])
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    # Get video comments
    def get_video_comments(self, video_id, max_results=100):
        comments = []
        next_page_token = None

        while len(comments) < max_results:
            comment_request = self.youtube.commentThreads().list(
                part='snippet', 
                videoId=video_id, 
                maxResults=100, 
                pageToken=next_page_token
            )
            comment_response = comment_request.execute()

            for item in comment_response['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])

            next_page_token = comment_response.get('nextPageToken')
            if not next_page_token:
                break

        return comments[:max_results]

    # Sentiment analysis of comments
    def analyze_sentiment(self, comments):
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for comment in comments:
            analysis = TextBlob(comment)
            if analysis.sentiment.polarity > 0:
                positive_count += 1
            elif analysis.sentiment.polarity < 0:
                negative_count += 1
            else:
                neutral_count += 1

        sentiment_results = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }

        plt.figure(figsize=(8, 5))
        plt.bar(sentiment_results.keys(), sentiment_results.values(), color=['green', 'red', 'gray'])
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Analysis of Comments')
        plt.show()

        return sentiment_results

    # Comments' word cloud
    def generate_word_cloud(self, comments):
        text = ' '.join(comments)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    # Download image through url
    def download_image(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        return None

    # Extract colors of thumbnails
    def get_dominant_colors(self, image, k=3):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(img)

        # Get RGB values of the cluster centers
        colors = kmeans.cluster_centers_.astype(int)
        background_color = colors[0]  
        foreground_color = colors[1]  
        return colors, background_color, foreground_color

    # Calculate colors' contrast
    def calculate_contrast(self, color1, color2):
        return distance.euclidean(color1, color2)

    def process_thumbnails_with_contrast(self, df, k=3):
        contrast_list = []
        for index, row in df.iterrows():
            url = row['thumbnail_url']
            img = self.download_image(url)

            if img is not None:
                _, background_color, foreground_color = self.get_dominant_colors(img, k)
                contrast = self.calculate_contrast(background_color, foreground_color)
                contrast_list.append(contrast)
            else:
                contrast_list.append(None)
        return contrast_list

    # Save thumbnail analysis to csv
    def save_thumbnail_analysis(self, df, output_file='thumbnail_analysis.csv', k=3):
        dominant_colors_list = []
        for index, row in df.iterrows():
            url = row['thumbnail_url']
            img = self.download_image(url)
            if img is not None:
                dominant_colors, _, _ = self.get_dominant_colors(img, k)
                flat_colors = dominant_colors.flatten()  
                dominant_colors_list.append(flat_colors)
            else:
                dominant_colors_list.append([None] * (k * 3))  

        # Generate color column names
        color_columns = [f'{channel}{i + 1}' for i in range(k) for channel in ['R', 'G', 'B']]
        dominant_colors_df = pd.DataFrame(dominant_colors_list, columns=color_columns)

        # Add contrast to DataFrame
        contrast_list = self.process_thumbnails_with_contrast(df, k)
        dominant_colors_df['contrast'] = contrast_list

        # Combine original DataFrame with analysis
        combined_df = pd.concat([df, dominant_colors_df], axis=1)
        combined_df.to_csv(output_file, index=False)
        print(f"Thumbnail analysis saved to {output_file}")

    # Correlation between colors and view counts
    def correlation_analysis(self, df):
        correlation_matrix = df[['R1', 'G1', 'B1', 'R2', 'G2', 'B2', 'R3', 'G3', 'B3', 'contrast', 'view_count']].corr()
        print(correlation_matrix['view_count'])

        # Visualize
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation between Thumbnail Features and Views')
        plt.show()


# Unit Tests
class TestYouTubeDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = '' # replace with your api key
        cls.analyzer = YouTubeDataAnalyzer(cls.api_key)
        cls.channel_name = "MrBeast"

    def test_search_youtuber_channel_id(self):
        channel_id = self.analyzer.search_youtuber_channel_id(self.channel_name)
        self.assertIsInstance(channel_id, str)

    def test_get_videos(self):
        channel_id = self.analyzer.search_youtuber_channel_id(self.channel_name)
        video_ids = self.analyzer.get_videos(channel_id, max_results=5)
        self.assertTrue(len(video_ids) > 0)

    def test_get_video_details(self):
        channel_id = self.analyzer.search_youtuber_channel_id(self.channel_name)
        video_ids = self.analyzer.get_videos(channel_id, max_results=5)
        video_details = self.analyzer.get_video_details(video_ids)
        self.assertTrue(len(video_details) > 0)
        self.assertIn('title', video_details[0])

    def test_get_video_comments(self):
        channel_id = self.analyzer.search_youtuber_channel_id(self.channel_name)
        video_ids = self.analyzer.get_videos(channel_id, max_results=1)
        comments = self.analyzer.get_video_comments(video_ids[0], max_results=10)
        self.assertTrue(len(comments) > 0)

    
    # Thumbnail Analysis Tests
    @patch('requests.get')  # Corrected patch
    def test_download_image_success(self, mock_get):
        # Mock the HTTP response content to simulate a valid image
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'fake_image_data'
        mock_get.return_value = mock_response
        
        # Create a dummy image object
        img = Image.new('RGB', (10, 10))
        with patch('PIL.Image.open', return_value=img):
            result = self.analyzer.download_image('https://i.ytimg.com/vi/bn0Kh9c4Zv4/hqdefault.jpg')
            self.assertIsInstance(result, Image.Image)

    @patch('requests.get')  # Corrected patch
    def test_download_image_failure(self, mock_get):
        # Simulate a failed image download
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.analyzer.download_image('https://i.ytimg.com/vi/bn0Kh9c4Zv4/hqdefault.jpg')
        self.assertIsNone(result)
    
    
if __name__ == "__main__":
    
    API_KEY = "" #replace with your own api key
    CHANNEL_NAME = "MrBeast"
    MAX_RESULTS = 2000

    analyzer = YouTubeDataAnalyzer(API_KEY)

    
    
    # power analysis
    analyzer.power_analysis()

    # Retrieve the channel ID
    channel_id = analyzer.search_youtuber_channel_id(CHANNEL_NAME)
    print(f"Channel ID for {CHANNEL_NAME}: {channel_id}")

    # Retrieve video IDs
    video_ids = analyzer.get_videos(channel_id, MAX_RESULTS)
    print(f"Retrieved {len(video_ids)} video IDs.")

    

    # Save video details to a CSV file
    # Check if the CSV file already exists
    video_details_csv = "video_details.csv"
    if not os.path.exists(video_details_csv):
        # Save video details to a CSV file if it doesn't exist
        # Get video details
        video_details = analyzer.get_video_details(video_ids)
        print(f"Retrieved details for {len(video_details)} videos.")
        analyzer.save_to_csv(video_details, video_details_csv)
        print(f"Saved video details to {video_details_csv}")
    else:
        print(f"{video_details_csv} already exists, skipping saving.")
    

    # Analyze comments for the first video
    if video_ids:
        comments = analyzer.get_video_comments(video_ids[0], max_results=200)
        print(f"Retrieved {len(comments)} comments for video ID {video_ids[0]}.")

        # Perform sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(comments)
        print("Sentiment Analysis Results:", sentiment_results)

        # Generate a word cloud
        analyzer.generate_word_cloud(comments)


    if os.path.exists(video_details_csv):
        video_df = pd.read_csv(video_details_csv)

        # Perform thumbnail analysis and save results if the file doesn't already exist
        thumbnail_analysis_csv = 'thumbnail_analysis.csv'
        if not os.path.exists(thumbnail_analysis_csv):
            analyzer.save_thumbnail_analysis(video_df, output_file=thumbnail_analysis_csv, k=3)
            print(f"Saved thumbnail analysis to {thumbnail_analysis_csv}")
        else:
            print(f"{thumbnail_analysis_csv} already exists, skipping thumbnail analysis.")

        # Correlation analysis
        thumbnail_df = pd.read_csv(thumbnail_analysis_csv)
        analyzer.correlation_analysis(thumbnail_df)

    unittest.main()

