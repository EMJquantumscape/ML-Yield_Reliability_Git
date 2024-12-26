# %%
import pandas as pd
import genealogy

from qsdc.client import Client


def query_mass(sample_ids, label, conn):
    # filter not none values
    sample_ids = [x for x in sample_ids if x is not None]

    # if sample_ids is empty, return empty dataframe
    if len(sample_ids) == 0:
        df = pd.DataFrame(columns=["displayname", label])

    else:

        sample_str = (
            tuple(sample_ids) if (len(sample_ids) > 1) else f"('{sample_ids[0]}')"
        )

        q = f"""SELECT displayname, value AS '{label}'
                FROM measurement 
                LEFT JOIN device_structure ON measurement.iddevice = device_structure.iddevice
                WHERE (displayname IN {sample_str}) AND (label = '{label}')"""

        df = pd.read_sql(q, conn)

        # remove duplicate displaynames, keeping the last one
        df.drop_duplicates(subset=["displayname"], keep="last", inplace=True)

    # join sample_ids and df
    df = pd.merge(
        pd.DataFrame(sample_ids, columns=["displayname"]),
        df,
        how="left",
        on="displayname",
    )

    return df


# conn = Client().get_mysql_engine()
# query_mass('APD251AC-US00-12', "US mass post-UCT", conn)


# %%
def get_mass_data(sample_ids):

    genealogy_df = genealogy.get_genealogy_unitcell(sample_ids)

    qs_client = Client()
    conn = qs_client.get_mysql_engine()

    df_mass_US = query_mass(genealogy_df["US_id"].dropna(), "US mass post-UCT", conn)
    df_mass_US.rename(columns={"displayname": "US_id"}, inplace=True)

    df_mass_US_rel = query_mass(
        genealogy_df["US_id"].dropna(), "US mass post-reliability", conn
    )
    df_mass_US_rel.rename(columns={"displayname": "US_id"}, inplace=True)

    df_mass_US_pouch = query_mass(
        genealogy_df["US_id"].dropna(), "US mass post-pouching", conn
    )
    df_mass_US_pouch.rename(columns={"displayname": "US_id"}, inplace=True)

    df_mass_CS = query_mass(genealogy_df["CS_id"].dropna(), "CS mass", conn)
    df_mass_CS.rename(columns={"displayname": "CS_id"}, inplace=True)

    df_mass_DP00 = query_mass(
        genealogy_df["SA00_id"].dropna(), "Dispensed SA mass", conn
    )
    df_mass_DP00.rename(
        columns={"displayname": "SA00_id", "Dispensed SA mass": "DP00 mass"},
        inplace=True,
    )
    df_mass_DP01 = query_mass(
        genealogy_df["SA01_id"].dropna(), "Dispensed SA mass", conn
    )
    df_mass_DP01.rename(
        columns={"displayname": "SA01_id", "Dispensed SA mass": "DP01 mass"},
        inplace=True,
    )

    df_mass_SA00 = query_mass(genealogy_df["SA00_id"].dropna(), "SA mass", conn)
    df_mass_SA00.rename(
        columns={"displayname": "SA00_id", "SA mass": "SA00 mass"}, inplace=True
    )
    df_mass_SA01 = query_mass(genealogy_df["SA01_id"].dropna(), "SA mass", conn)
    df_mass_SA01.rename(
        columns={"displayname": "SA01_id", "SA mass": "SA01 mass"}, inplace=True
    )
    df_mass_CAT = query_mass(
        genealogy_df["CAT_id"].dropna(), "Cathode + substrate mass", conn
    )
    df_mass_CAT.rename(columns={"displayname": "CAT_id"}, inplace=True)

    # join all mass data into genealogy_df
    df_mass = pd.merge(genealogy_df, df_mass_US, how="left", on="US_id")
    df_mass = pd.merge(df_mass, df_mass_US_rel, how="left", on="US_id")
    df_mass = pd.merge(df_mass, df_mass_US_pouch, how="left", on="US_id")
    df_mass = pd.merge(df_mass, df_mass_CS, how="left", on="CS_id")
    df_mass = pd.merge(df_mass, df_mass_DP00, how="left", on="SA00_id")
    df_mass = pd.merge(df_mass, df_mass_DP01, how="left", on="SA01_id")
    df_mass = pd.merge(df_mass, df_mass_SA00, how="left", on="SA00_id")
    df_mass = pd.merge(df_mass, df_mass_SA01, how="left", on="SA01_id")
    df_mass = pd.merge(df_mass, df_mass_CAT, how="left", on="CAT_id")

    df_mass["process"] = df_mass["US_id"].str[0:8]
    df_mass["batch"] = df_mass["US_id"].str[0:13]

    # if DP00/DP01 is not NaN, but SA00/SA01 is NaN, set SA00/SA01 mass to DP00/DP01 mass - 30
    df_mass.loc[
        (~df_mass["DP00 mass"].isna()) & (df_mass["SA00 mass"].isna()), "SA00 mass"
    ] = (df_mass["DP00 mass"] - 30)
    df_mass.loc[
        (~df_mass["DP01 mass"].isna()) & (df_mass["SA01 mass"].isna()), "SA01 mass"
    ] = (df_mass["DP01 mass"] - 30)

    # if DP00/DP01 is not NaN, subtract SA00/SA01 mass from DP00 mass
    df_mass.loc[~df_mass["DP00 mass"].isna(), "DP00 mass"] = (
        df_mass["DP00 mass"] - df_mass["SA00 mass"]
    )
    df_mass.loc[~df_mass["DP01 mass"].isna(), "DP01 mass"] = (
        df_mass["DP01 mass"] - df_mass["SA01 mass"]
    )

    # if DP00/DP01 mass values are not between 25 and 50, set to NaN
    df_mass.loc[
        (df_mass["DP00 mass"] < 20) | (df_mass["DP00 mass"] > 70), "DP00 mass"
    ] = None
    df_mass.loc[
        (df_mass["DP01 mass"] < 20) | (df_mass["DP01 mass"] > 70), "DP01 mass"
    ] = None

    # fill DP00 and DP01 NaN values with average value for that process and if that doesn't exist, use 30 mg
    df_mass["DP00 mass"] = df_mass.groupby("process")["DP00 mass"].apply(
        lambda x: x.fillna(x.mean())
    )
    df_mass["DP00 mass"] = df_mass["DP00 mass"].fillna(30)
    df_mass["DP01 mass"] = df_mass.groupby("process")["DP01 mass"].apply(
        lambda x: x.fillna(x.mean())
    )
    df_mass["DP01 mass"] = df_mass["DP01 mass"].fillna(30)

    df_mass["CS ESS mass"] = (
        df_mass["CS mass"]
        - df_mass["DP00 mass"]
        - df_mass["DP01 mass"]
        - df_mass["SA00 mass"]
        - df_mass["SA01 mass"]
        - df_mass["Cathode + substrate mass"]
    )
    df_mass["US ESS mass"] = (
        df_mass["US mass post-UCT"]
        - df_mass["DP00 mass"]
        - df_mass["DP01 mass"]
        - df_mass["SA00 mass"]
        - df_mass["SA01 mass"]
        - df_mass["Cathode + substrate mass"]
    )
    df_mass["CS to US mass loss"] = df_mass["CS mass"] - df_mass["US mass post-UCT"]

    df_mass["US to reliability mass loss"] = (
        df_mass["US mass post-UCT"] - df_mass["US mass post-reliability"]
    )
    df_mass["US to pouching mass loss"] = (
        df_mass["US mass post-UCT"] - df_mass["US mass post-pouching"] - 114.3
    )

    # remove duplicate rows
    df_mass = df_mass.drop_duplicates(subset=["US_id"], keep="first")

    return df_mass


# %%
# test = get_mass_data([])


# %%


def get_dispense_mass(batches):
    # batches = ["APD215AB"]
    batches = "|".join(batches)
    qs_client = Client()
    conn = qs_client.get_mysql_engine()

    q = f"""SELECT displayname, label, value AS 'mass', measurement_timestamp AS 'meas_time'
            FROM measurement 
            LEFT JOIN device_structure ON measurement.iddevice = device_structure.iddevice
            WHERE displayname REGEXP '{batches}' 
            AND label IN ('SA mass','Dispensed SA mass')"""

    df = pd.read_sql(q, conn)

    # capitalize first letter of label
    df["label"] = df["label"].str.upper()

    # pivot table around label
    df_pivot = df.pivot_table(
        index="displayname", columns="label", values="mass", aggfunc="last"
    )

    # merge meas_time
    df_pivot = df_pivot.merge(
        df[["displayname", "meas_time"]].drop_duplicates(keep="last"),
        how="left",
        on="displayname",
    )

    # drop rows with mass values NaN
    df_pivot = df_pivot.dropna(subset=["SA MASS", "DISPENSED SA MASS"])

    df_pivot["Dispense mass"] = df_pivot["DISPENSED SA MASS"] - df_pivot["SA MASS"]

    return df_pivot


# %%
