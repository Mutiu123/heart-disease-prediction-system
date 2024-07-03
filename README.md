## **Project Description**

The developed **AI-Based Movies Recommender System** aims to enhance user satisfaction by providing personalised movie recommendations. Here's a more comprehensive overview:

- **Objective**: The system recommends relevant movies to users based on their preferences, viewing history, and other relevant factors.
- **Audience**: Movie enthusiasts, streaming platform users, and anyone seeking entertainment options.
- **Purpose**:
  - Facilitate content discovery.
  - Improve user engagement and retention.
  - Enhance the overall movie-watching experience.

## **Problem Statement**

The central problem the system addresses is how to recommend movies effectively. Consider the following challenges:

1. **Cold Start Problem**:
   - New users have limited data, making personalized recommendations challenging.
   - How can your system handle this initial lack of user history?

2. **Data Sparsity**:
   - User-item interaction matrices are often sparse due to limited ratings or interactions.
   - How do you handle missing data while still providing meaningful recommendations?

3. **Scalability**:
   - As the user base grows, computation efficiency becomes crucial.
   - How can your system handle large datasets without sacrificing performance?

4. **Accuracy vs. Diversity**:
   - Striking a balance between recommending popular movies (accuracy) and introducing users to lesser-known gems (diversity).

## **Methodology**

The methodology involves a blend of recommendation techniques:

1. **Collaborative Filtering (CF)**:
   - **User-Based CF**: Identifies users with similar movie preferences based on their historical ratings.
   - **Item-Based CF**: Finds movies similar to the ones a user has liked.
   - **Matrix Factorization**: Decomposes the user-item matrix to discover latent features.

2. **Content-Based Filtering**:
   - Analyses movie features (genres, cast, directors, plot keywords) to make recommendations.
   - Utilizes techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings.

3. **Hybrid Approaches**:
   - Combines collaborative and content-based methods for robust recommendations.
   - Weighted blending or parallel execution can enhance accuracy.

4. **Data Preprocessing**:
    - **Data Collection**: Gathers movie metadata from sources like IMDb, streaming platforms, or the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). 
   - **Data Cleaning**: Removes duplicates, handles null values, and standardizes formats.
   - **Feature Extraction**: Creates relevant features (e.g., bag-of-words representations).

## **Project Modules**

1. **Data Collection and Preprocessing**:
   - **Web Scraping**: Extracts movie details (titles, genres, release years) from APIs or websites.
   - **Data Cleaning**: Removes duplicates, handles null values, and standardizes formats.
   - **User Profiles**: Creates user profiles based on viewing history.

2. **Feature Engineering**:
   - **Movie Features**: Extracts genres, cast, directors, release years, and plot keywords.
   - **User Features**: Captures user preferences, watch history, and demographic information.

3. **Model Training and Selection**:
   - **Collaborative Filtering Models**: Trains user-based and item-based CF models.
   - **Content-Based Models**: Utilizes TF-IDF, word embeddings, or even neural networks.
   - **Hybrid Models**: Combines recommendations from both approaches.

4. **Recommendation Generation**:
   - **Personalized Recommendations**: Generates top-N movie lists for each user.
   - **Real-Time Scoring**: Efficiently computes recommendations during user interactions.

5. **Evaluation**:
   - Assesses recommendation quality using metrics like RMSE (Root Mean Squared Error), precision, recall, and F1-score.
   - Conducts A/B testing to validate improvements against existing recommendation methods.

## **Contribution**

Highlight of the  unique contributions:
- Enhance existing methods?
- Address specific challenges (e.g., cold start, scalability) in a novel way.

## **Applications**

The broader impact of the system:
- **User Experience Enhancement**: Users discover movies aligned with their tastes.
- **Platform Retention**: Improves user engagement and reduces churn.
- **Business Monetization**: Recommendations drive content consumption and potential affiliate marketing.

## **Future Directions**

Explore these avenues for further enhancement:

1. **Deep Learning Models**:
   - Investigate neural collaborative filtering or recurrent neural networks.
   - Capture complex user-item interactions.

2. **Temporal Dynamics**:
   - Account for changing preferences over time (seasonal trends, evolving tastes).

3. **Cold Start Solutions**:
   - Develop strategies for new users and items (e.g., content-based recommendations initially).

4. **Interpretability**:
   - Explain why certain movies are recommended (e.g., attention mechanisms).

5. **User Feedback Integration**:
   - Continuously refine recommendations based on user feedback.

## **Below are System Demo:**

![The System Demo](https://github.com/Mutiu123/movie-recommender-system/blob/main/demo_image/demo.png)

![The System Demo](https://github.com/Mutiu123/movie-recommender-system/blob/main/demo_image/demo2.png)


## **To run the model**
1. **Clone the Repository**:
   - First, clone the repository containing your movie recommendation system code to your local machine. You can do this using Git or by downloading the ZIP file from the repository.

2. **Install Dependencies**:
   - Open your terminal or command prompt and navigate to the project directory.
   - Install the necessary Python libraries mentioned in the `requirements.txt` file using the following command:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Streamlit App**:
   - In the same terminal or command prompt, execute the following command to run the Streamlit app:
     ```
     streamlit run app.py
     ```
   - This will start the local development server, and you'll see a message indicating that the app is running.
   - Open your web browser and visit `http://localhost:8501` (or the URL provided in the terminal) to access the interactive web app.

4. **Select a Movie**:
   - On the Streamlit app, you'll find a search bar where you can either select a movie from the dropdown list or type the movie name.
   - Choose a movie for which you want to receive recommendations.

5. **View Recommendations**:
   - Click the "Show Recommendation" button.
   - The app will display recommended movies based on the selected input movie. You'll see movie posters and titles for the top recommendations.

6. **Explore Further**:
   - Feel free to explore other movies by selecting different titles or typing new ones.
   - The app dynamically updates recommendations based on your input.

