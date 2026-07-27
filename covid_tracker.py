
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "owid-covid-data.csv"

def load_data():
    df = pd.read_csv(CSV_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df

def country_data(df, country):
    c = df[df["location"] == country].copy()
    c = c[["date","new_cases","total_cases"]].fillna(0)
    c["Rolling_Avg"] = c["new_cases"].rolling(7).mean()
    c["Growth_Rate_%"] = np.where(
        c["new_cases"].shift(1) > 0,
        ((c["new_cases"] - c["new_cases"].shift(1))
         / c["new_cases"].shift(1)) * 100,
        0
    )
    return c

def detect_peak(c):
    idx = c["new_cases"].idxmax()
    return c.loc[idx,"date"], c.loc[idx,"new_cases"]

def plot_country(c,country):
    plt.figure(figsize=(10,5))
    plt.plot(c["date"],c["new_cases"],label="Daily Cases")
    plt.plot(c["date"],c["Rolling_Avg"],label="7-Day Rolling Avg")
    plt.title(f"COVID-19 Cases - {country}")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{country}_cases.png")
    plt.show()

def compare(df,countries):
    plt.figure(figsize=(10,5))
    for country in countries:
        c = country_data(df,country)
        plt.plot(c["date"],c["Rolling_Avg"],label=country)
    plt.title("Country Comparison")
    plt.xlabel("Date")
    plt.ylabel("7-Day Rolling Average")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()

def export_summary(country,c,peak_date,peak_cases):
    summary = {
        "Country":country,
        "Peak Date":str(peak_date.date()),
        "Peak Cases":int(peak_cases),
        "Average Daily Cases":round(c["new_cases"].mean(),2),
        "Total Cases":int(c["total_cases"].max())
    }
    with open("summary.txt","w") as f:
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
    print("Summary exported to summary.txt")

def main():
    df = load_data()
    print("Available example countries: Pakistan, India, United States")
    country = input("Enter country name: ")
    c = country_data(df,country)
    if c.empty:
        print("Country not found.")
        return
    peak_date, peak_cases = detect_peak(c)
    print(f"Peak Date : {peak_date.date()}")
    print(f"Peak Cases: {int(peak_cases)}")
    plot_country(c,country)
    compare(df,[country,"India","United States"])
    export_summary(country,c,peak_date,peak_cases)

if __name__ == "__main__":
    main()
