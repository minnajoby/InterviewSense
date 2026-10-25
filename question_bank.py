
# Interview questions

def get_categories():
    return [
        "HR Interview",
        "Technical Interview",
        "Data Science Interview"
    ]


def get_questions(category):
    questions = {
        "HR Interview": [
            "Tell me about yourself.",
            "What are your strengths and weaknesses?",
            "Why should we hire you?",
            "Describe a challenge you faced and how you handled it.",
            "Where do you see yourself in five years?"
        ],

        "Technical Interview": [
            "Explain a project you have worked on.",
            "What is the difference between a list and a tuple in Python?",
            "What is the difference between SQL and Python?",
            "What is overfitting in machine learning?",
            "How do you handle missing values in a dataset?"
        ],

        "Data Science Interview": [
            "What is the difference between supervised and unsupervised learning?",
            "Explain the purpose of train-test splitting.",
            "What is the difference between classification and regression?",
            "What is feature selection?",
            "How would you evaluate a classification model?"
        ]
    }

    return questions.get(category, questions["HR Interview"])

