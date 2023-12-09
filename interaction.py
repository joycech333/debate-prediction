import sentiment
import util
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.api import OLS
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def process_debate():
    years = []
    full_paths = [f'scraped-data/transcripts/{file}' for file in util.FILES]
    for file_path in full_paths:
        file_date = file_path[25:]
        year = file_date.split('_')[2][:-4]

        all_speakers = util.split_speakers(file_path)

        for speaker in all_speakers:
            debate_cands = util.PARTICIPANTS[file_date]

            # this means the speaker was the moderator
            if speaker not in debate_cands:
                continue

            years.append(year)

    return years

def make_csv():
    full_paths = [f'scraped-data/transcripts/{file}' for file in util.FILES]

    # X, y = sentiment.process_debate(full_paths)
    # Xdf = pd.DataFrame(X)
    # ydf = pd.DataFrame(y)
    # Xdf = pd.read_csv('xys/sentiments.csv')
    # ydf = pd.read_csv('xys/sentiment_ys.csv')
    # Xdf.to_csv('xys/sentiments.csv', index=False)
    # ydf.to_csv('xys/sentiment_ys.csv', index=False)

    # years = []
    # # get year column
    # for file in util.FILES:
    #     years.append(file.split('_')[2][:-4])
    # years = process_debate()

    # Xdf['year'] = years

    # print(Xdf.info())
    # Xdf.to_csv('xys/sentiments.csv', index=False)
    

if __name__ == "__main__":
    X = pd.read_csv('xys/sentiments.csv')
    y = pd.read_csv('xys/sentiment_ys.csv')

    X['y'] = y

    plt.figure(figsize=(6, 6))
    sns.lmplot(data=X, x="sentiment", y="y", logistic=True, ci=None)
    plt.title(f'Logistic Regression on Sentiment')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Winningness')
    plt.savefig('xys/sentimentlr.png', bbox_inches="tight")

    logisticRegr = LogisticRegressionCV(cv=10, random_state=0)
    y_pred = cross_val_predict(logisticRegr, np.reshape(X['sentiment'], (-1, 1)), y.values.ravel(), cv=10) # StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    print('Average Accuracy:', accuracy)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Lose', 'Predicted Win'], yticklabels=['True Lose', 'True Win'])
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'xys/sentimentlrmx.png')

    """
    print(X)

    print('renaming')
    X.rename(columns={0: "sentiment"}, inplace=True)
    print(X)

    # normalize
    # X['sentiment'] = (X['sentiment'] - X['sentiment'].mean()) / (X['sentiment'].std())
    # X['year'] = (X['year'] - X['year'].mean()) / (X['year'].std())
    X['y'] = y

    print(X)

    sns.lmplot(x='0', y='y', data=X, hue='year')
    plt.ylim((-0.1, 1.1))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # X_train['y'] = y_train
    # sns.pointplot(x='sentiment', y='y', data=X_train, hue='year')
    # plt.show()

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_inter_const = sm.add_constant(X_train_poly)

    # model = LogisticRegression()
    # model.fit(X_train_poly, y_train)

    # X_test_poly = poly.transform(X_test)
    # y_pred = model.predict(X_test_poly)
    
    # accuracy = accuracy_score(y_test, y_pred)
    # print('interaction', accuracy)

    print(X_inter_const)
    interAll = pd.DataFrame(X_inter_const)
    interAll['y'] = y_train
    
    interAll.rename(columns={1: "sentiment", 2: "year", 3: "interaction"}, inplace=True)
    print(interAll)

    # model = OLS(y_train, X_inter_const).fit()
    # print(model.summary())


    sns.lmplot(data=interAll, x="sentiment", y="y", hue="year")
    plt.show()

    # ORIGINAL MODEL
    # print('\n ORIGINAL WITHOUT INTERACTION:')
    # X_train_orig = sm.add_constant(X_train['sentiment'])
    # model = OLS(y_train, X_train_orig).fit()
    # print(model.summary())

    # X_train['y'] = y_train
    # sns.regplot(x='sentiment', y='y', data=X_train) # , logistic=True
    # plt.savefig('sentimentplt.png')

    # interaction_plot(weight, duration, days, colors=['red','blue'], markers=['D','^'], ms=10)

    # df = pd.DataFrame()
    # df['sentiment'] = X_train.iloc[:,0] # sentiment
    # df['year'] = X_train['year']
    # df['y'] = y_train # ys
    # model = ols('y ~ C(sentiment) + C(year) + C(sentiment):C(year)', data=df).fit()
    # print(sm.stats.anova_lm(model, typ=2))
    


    # ORIGINAL MODEL
    # X_train, X_test, y_train, y_test = train_test_split(np.reshape(X.iloc[:,0], (-1, 1)), y, test_size=0.25, random_state=0)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)

    # print('original', accuracy)

    """