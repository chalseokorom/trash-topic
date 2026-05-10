"""Create data visualzations based on review data and BERTopic model output"""
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def get_star_review_plot(reviews: pd.DataFrame) -> None:
    """Create pie plot of star ratings of text reviews"""
    if Path("img/star_reviews_pie_plot.svg").exists():
        return

    review_star_percentages = reviews.stars.value_counts() / len(reviews)
    review_star_percentages = (
        review_star_percentages * 100).round(2).sort_index()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect="equal"))

    star_labels = [str(r[0] + 1) + ' $\U00002605$ - ' + str(r[1]) +
                   "%" for r in enumerate(review_star_percentages)]
    ax.pie(review_star_percentages, labels=star_labels,
           wedgeprops=dict(width=0.5), startangle=-40)

    ax.set_title(
        """Are customers more likely to leave a text review 
            when they are angry (1-star) or happy (5-star)?""")

    plt.savefig("img/star_reviews_pie_plot.svg")
    plt.close()
    fig.clear()


def get_cumulative_review_plot(text_reviews: pd.DataFrame, star_reviews: pd.DataFrame) -> None:
    """Create a animated histogram showing star ratings from all reviews over time"""
    if Path("img/cumulative_reviews_animation.gif").exists():
        return

    from matplotlib.animation import FuncAnimation, PillowWriter

    text_reviews, star_reviews = text_reviews.copy(), star_reviews.copy()
    text_reviews['has_text'], star_reviews['has_text'] = True, False

    df = pd.concat([text_reviews, star_reviews])
    df['publishedAtDate'] = pd.to_datetime(
        df['publishedAtDate'], format='ISO8601', utc=True)
    df = df.sort_values('publishedAtDate')

    fig, ax = plt.subplots(figsize=(10, 6))
    frames = pd.date_range(df['publishedAtDate'].min(),
                           df['publishedAtDate'].max(), freq='ME')

    final_max = df.groupby(['stars', 'has_text']).size().unstack(
        fill_value=0).sum(axis=1).max()
    y_limit = final_max * 1.15

    def add_animation(current_date):
        """Update the animation with new information as the date changes"""
        ax.clear()

        subset = df[df['publishedAtDate'] <= current_date]
        counts = subset.groupby(['stars', 'has_text']
                                ).size().unstack(fill_value=0)
        counts = counts.reindex(index=[1, 2, 3, 4, 5], columns=[
                                True, False], fill_value=0)

        x = np.arange(1, 6)
        width = 0.35

        ax.bar(x - width/2, counts[True], width,
               label='With Text', color='#3498db')
        ax.bar(x + width/2, counts[False], width,
               label='Without Text', color='#e74c3c')

        ax.set_ylim(0, y_limit)
        ax.set_xticks(x)
        ax.set_xlabel('Star Rating')
        ax.set_ylabel('Total Reviews')
        ax.set_title(
            f'Cumulative Reviews up to {current_date.strftime("%B %Y")}')
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    ani = FuncAnimation(fig, add_animation, frames=frames, interval=200)
    ani.save('img/cumulative_reviews_animation.gif',
             writer=PillowWriter(fps=5))

    plt.close(fig)
