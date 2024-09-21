from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training Data (feedback with labels)
train_feedback = [
    "I love this product, it’s amazing!",            # Useful (1)
    "Great service, will use again.",                # Useful (1)
    "Buy now and get 50% off! Limited time only!",   # Spam (0)
    "Click here to claim your free prize!",          # Spam (0)
    "The customer support was really helpful.",      # Useful (1)
    "Spam message: win a million dollars now!",      # Spam (0)
    "I’m not satisfied with the quality of the item.", # Useful (1)
    "Get rich fast with this one simple trick!",     # Spam (0)
    "Product arrived late, but still happy.",        # Useful (1)
    "Congratulations! You've won a free iPhone.",    # Spam (0)
    "Excellent app, very easy to use.",              # Useful (1)
    "This is not what I ordered, very disappointed.",# Useful (1)
    "Earn $1000 from home, no experience needed.",   # Spam (0)
    "Thank you for resolving my issue so quickly!",  # Useful (1)
    "Order today and get a special discount!",       # Spam (0)
    "Really satisfied with the overall service.",    # Useful (1)
    "This product changed my life!",                 # Useful (1)
    "Don’t miss out on our special offer!",          # Spam (0)
    "Highly recommend this product to everyone.",    # Useful (1)
    "Terrible experience, would not buy again.",     # Useful (1),
    "You’ve been selected for a limited-time offer!",# Spam (0)
    "Free vacation just for signing up!",            # Spam (0)
    "The item broke after a week of use.",           # Useful (1)
    "Excellent customer service, very satisfied.",   # Useful (1),
    "Avoid this product, it’s a scam.",              # Useful (1),
    "Winner! Claim your gift card now.",             # Spam (0),
    "Get instant access to exclusive deals!",        # Spam (0),
    "The food at the restaurant was amazing!",       # Useful (1),
    "I received a broken product, horrible service.",# Useful (1),
    "Cheap deals, click here for more info.",        # Spam (0),
    "Fantastic value for the price!",                # Useful (1)
]
train_labels = [
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
    1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
    0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
    1
] # 1 = useful, 0 = spam

# Prediction Data (feedback without labels)
predict_feedback = ["a valuable feedback i got"]
# Step 1: Preprocess and Vectorize the Training Data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_feedback)

# Step 2: Train the Model on the Training Data
model = MultinomialNB()
model.fit(X_train_tfidf, train_labels)

# Step 3: Preprocess and Vectorize the Independent Prediction Data
X_predict_tfidf = vectorizer.transform(predict_feedback)

# Step 4: Make Predictions on the Independent Prediction Data
predictions = model.predict(X_predict_tfidf)

print(f"predictions table: {predictions}")

# Print the feedback and the corresponding predictions
for feedback, prediction in zip(predict_feedback, predictions):
    label = "Useful" if prediction == 1 else "Spam"
    print(f"Feedback: '{feedback}' -> Predicted as: {label}")
