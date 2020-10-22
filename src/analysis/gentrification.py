def get_eligibility(df):
    df['eligibility'] = (
            df['population'] > 500
            and df['household_income'] < df['household_income'].quantile(.4)
            and df['home_value'] < df['home_value'].quantile(.4)
    )
