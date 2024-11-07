# %%
#Import libraries and functions
import pandas as pd
import numpy as np
import datetime as dt
import genealogy_v2
import unit_cell_metro_metrics_ZI
import matplotlib.pyplot as plt
import sys
from lifelines import KaplanMeierFitter
from statsmodels.stats.proportion import proportion_confint
from lifelines.utils import restricted_mean_survival_time
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
import plotly
pio.templates.default = "plotly_white"
clrs = plotly.colors.DEFAULT_PLOTLY_COLORS
from qsdc.client import Client
import met_client as app
#import mass
import unit_cell_electrical_yield_and_metrics_with_rel as uceym_rel
import logging
from typing import List
import qs.et.etp_pb2 as etp_pb2
from datetime import datetime
from met_client import SearchQuery, ImageAgent
from met_client.constants import AnalysisType, ImageSize
from image_client.client import ImageClient
from image_client.manual_review import convert_manual_reviews_to_dataframe

qs_data_client = Client()
# create our data client
qs_client = Client()

# This is the experiment code we want to look at. Default:
#search = "APD256|MLB|MLD|QSC"
search = "MLB|MLD"
exclude = "None"


# Yield criteria

screen_softstart1C_recipes = [15691, 15696]
softstart1C_charge_capacity_fraction = 0.95
softstart1C_dvdt = -4.0
softstart1C_delta_dvdt = 2
softstart1C_CE = 0.98
softstart1C_ceiling_hold_time = 3600

screen_fastcharge = [14445, 15697]
fastcharge_charge_capacity_fraction = 0.95
fastcharge_dvdt = -10
fastcharge_delta_dvdt = 2
fastcharge_CE = 0.98
fastcharge_ceiling_hold_time = 3600

screen_Co3 = [13708, 15618, 14446]  # , 13213,13197, 13708, 13345, ]
Co3_charge_capacity = 202
Co3_dvdt = -10
Co3_charge_capacity_fraction = 1.04
Co3_charge_capacity_fraction_cycle = 1.01

reliability_recipes = [13720, 13706, 14398, 14633, 14645, 14599]
reliability_charge_capacity = 195
reliability_dvdt = -50

alct_reliability_recipes = [14444, 14745, 14781]
alct_reliability_charge_capacity = 240
alct_reliability_dvdt = -20


lowtemp_reliability_recipes = [14654]
lowtemp_reliability_charge_capacity = 180
lowtemp_reliability_dvdt = -5


# Query data
conn = qs_client.get_mysql_engine()

recipes = "|".join(
    [
        str(x)
        for x in (
            screen_Co3
            # + screen_Co3_RPT
            + screen_softstart1C_recipes
            + screen_fastcharge
           
            # + reliability_recipes
            # + alct_reliability_recipes
            # + reliability_track_cycle_charge
        )
    ]
)

df_raw = pd.read_sql_query(
    """
SELECT
  device_structure.displayname AS samplename,
  test_run_E12_cycle.VoltagePostCeilingRestEndDVdt * 1E6 AS dvdt,
  test_run_E12_cycle.CapacityChargeActiveMassSpecific AS 'AMSChargeCapacity',
  test_run_E12_cycle.CapacityDischargeActiveMassSpecific AS 'AMSDischargeCapacity',
  test_run_E12_cycle.CapacityDischarge AS 'DischargeCapacity',
  test_run_E12_cycle.CapacityCharge AS 'ChargeCapacity',
  test_run_E12_cycle.EnergyDischarge AS 'DischargeEnergy',
  test_run_E12_cycle.CapacityChargeFraction AS 'ChargeCapacityFraction',
  test_run_E12_cycle.CoulombicEfficiency AS 'CE',
  test_run_E12_cycle.AsrDcChargeMedian AS 'MedChargeASR',
  test_run_E12_cycle.AsrDcDischargeMedian AS 'MedDischargeASR',
  (test_run_E12_cycle.AsrDcChargeMedian/test_run_E12_cycle.AsrDcDischargeMedian) AS 'ASR_ratio',
  test_run_E12_cycle.TimeCeilingHold AS 'CeilingHoldTime',
  test_run_E12_cycle.VoltageEndCeilingRest AS 'CeilingRestVoltage',
  test_run_E12_cycle.`index` AS 'CycleIndex',
  test_run.`Index` AS 'RunIndex',
  test_run.idtest_recipe,
  test_run_E12_cycle.datetime_start AS 'TestCycleStart',
  test_run_E12_cycle.datetime_end AS 'TestCycleEnd',
  test_run_E12_cycle.IsShorted AS 'HardShort',
  test_run_E12_cycle.idtest_run_E12_cycle,
  test_run_E12.ProcessorAssumedCapacity_mAh AS 'ProcessorAssumedCapacity',
  test_run_E12.ocv_initial AS 'OCVInitial',
  process_flow.description AS 'ProcessDescription',
  process.started AS 'cell_build_time',
  tool.displayname AS Tool,
  test_run.Channel
FROM test_run_E12_cycle
  INNER JOIN test_run_E12 ON test_run_E12_cycle.idtest_run_E12 = test_run_E12.idtest_run_E12
  INNER JOIN test_run ON test_run_E12.idtest_run = test_run.idtest_run
  INNER JOIN test_setup_E12 ON test_run_E12.idtest_setup_E12 = test_setup_E12.idtest_setup_E12
  INNER JOIN test_request ON test_run.idtest_request = test_request.idtest_request
  INNER JOIN device_structure ON test_run.iddevice = device_structure.iddevice
  INNER JOIN process ON device_structure.idprocess_createdby = process.idprocess
  INNER JOIN process_flow ON process_flow.idprocess_flow = process.idprocess_flow
  INNER JOIN tool ON test_run.idtool=tool.idtool
WHERE 
device_structure.displayname REGEXP %(search)s
AND test_run_E12_cycle.CapacityCharge > 1
AND NOT device_structure.displayname REGEXP %(exclude)s
AND test_run.idtest_recipe REGEXP (%(recipe)s)

""",
    conn,
    params={"search": search, "recipe": recipes, "exclude": exclude},
)

df_raw = df_raw.sort_values(["RunIndex", "CycleIndex"], ascending=True)

# Determine if the cycle was stopped on short (more efficient method than to query the database)
df_raw["CumulativeCycle"] = 1
df_raw.CumulativeCycle = df_raw.groupby("samplename").CumulativeCycle.cumsum()

df_raw["CumulativeCycle_Rel"] = 1
# df_raw.loc[
#     df_raw["idtest_recipe"].isin(reliability_recipes + alct_reliability_recipes),
#     "CumulativeCycle_Rel",
# ] = (
#     df_raw[df_raw["idtest_recipe"].isin(reliability_recipes + alct_reliability_recipes)]
#     .groupby("samplename")
#     .CumulativeCycle_Rel.cumsum()
# )

df_raw.reset_index(inplace=True)

df_raw["last_cycle"] = (
    df_raw.groupby("samplename")["CumulativeCycle"].transform(max)
    == df_raw["CumulativeCycle"]
)

df_raw["StoppedOnShort"] = (
    df_raw["DischargeCapacity"].isnull()
    & df_raw["last_cycle"]
    & df_raw["TestCycleEnd"].notnull()
)

# df_raw.to_csv("df_raw.csv", index=False)


# %%
# ===========================================================================================
# ======================        CYCLE METRICS CALCULATION          ==========================
# ===========================================================================================

df_cyc = df_raw.copy()

df_cyc["batch"] = df_cyc["samplename"].str.slice(stop=13)
df_cyc["process"] = df_cyc["samplename"].str.slice(stop=8)
df_cyc["experiment"] = df_cyc["samplename"].str.slice(stop=6)
df_cyc["project"] = df_cyc["samplename"].str.slice(stop=3)

# group by sample and check if alct_reliability_recipes is in idtest_recipe
df_cyc["alct_test"] = df_cyc.groupby("batch")["idtest_recipe"].transform(
    lambda x: x.isin(alct_reliability_recipes).any()
)

df_cyc = df_cyc.set_index("samplename")


df_cyc["AMSDischargeCapactiy_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["AMSDischargeCapacity"]
    ]
    .groupby("samplename")
    .min()
)

df_cyc["DischargeCapactiy_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["DischargeCapacity"]
    ]
    .groupby("samplename")
    .min()
)

df_cyc["ChargeCapacity_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["ChargeCapacity"]
    ]
    .groupby("samplename")
    .max()
)

df_cyc["AMSDischargeCapactiy_Co3"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["AMSDischargeCapacity"]]
    .groupby("samplename")
    .min()
)

df_cyc["DischargeCapactiy_Co3"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeCapacity"]]
    .groupby("samplename")
    .min()
)

df_cyc["MedDischargeASR_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["MedDischargeASR"]
    ]
    .groupby("samplename")
    .last()
)


df_cyc["dVdt_delta_1C"] = np.abs(
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        ["dvdt"],
    ]
    .groupby("samplename")
    .min()
    - df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        ["dvdt"],
    ]
    .groupby("samplename")
    .max()
)

df_cyc["dVdt_delta_fastcharge"] = np.abs(
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_fastcharge + screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        ["dvdt"],
    ]
    .groupby("samplename")
    .min()
    - df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_fastcharge + screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        ["dvdt"],
    ]
    .groupby("samplename")
    .max()
)


df_cyc["dVdt_1C"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["dvdt"]]
    .groupby("samplename")
    .min()
)


df_cyc["CeilingHoldTime_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes),
        ["CeilingHoldTime"],
    ]
    .groupby("samplename")
    .last()
)

df_cyc["CE_1C"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["CE"]]
    .groupby("samplename")
    .last()
)


df_cyc = df_cyc.reset_index()

# If it stopped on short, it failed
df_cyc["Failed"] = df_cyc["StoppedOnShort"] == 1
df_cyc["Failed_reliability"] = df_cyc["StoppedOnShort"] == 1


# Soft-start 1C-1C
# 1C-1C
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(screen_softstart1C_recipes))
    & (
        (
            # (df_cyc.AMSChargeCapacity > softstart1C_charge_capacity)
            (df_cyc.ChargeCapacityFraction > softstart1C_charge_capacity_fraction)
            # | (df_cyc.AMSChargeCapacity < 50)
        )
    #     | (df_cyc.dvdt <= softstart1C_dvdt)
    #     | ((df_cyc.CE < softstart1C_CE) & (df_cyc.AMSDischargeCapacity > 140))
    #     | (df_cyc.dVdt_delta_1C > softstart1C_delta_dvdt)
    #     | (df_cyc.CeilingHoldTime > softstart1C_ceiling_hold_time)
    ),
    "Failed",
] = True

# Fast charge
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(screen_fastcharge))
    & (
        (
            (df_cyc.ChargeCapacityFraction > fastcharge_charge_capacity_fraction)
            # | (df_cyc.AMSChargeCapacity < 50)
        )
        | (df_cyc.dvdt <= fastcharge_dvdt)
        # | ((df_cyc.CE < fastcharge_CE) & (df_cyc.AMSDischargeCapacity > 140))
        # | (df_cyc.dVdt_delta_fastcharge > fastcharge_delta_dvdt)
        # | (df_cyc.CeilingHoldTime > fastcharge_ceiling_hold_time)
    ),
    "Failed",
] = True

# C/3 cycle
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(screen_Co3))
    & (
        (
            (df_cyc.AMSChargeCapacity > Co3_charge_capacity)
            # | (df_cyc.AMSChargeCapacity < 100)
        )
        | (df_cyc.dvdt <= Co3_dvdt)
    ),
    "Failed",
] = True



# Reliability
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(reliability_recipes))
    & (
        (df_cyc.AMSChargeCapacity > reliability_charge_capacity)
        | (df_cyc.dvdt <= reliability_dvdt)
    ),
    "Failed_reliability",
] = True

# ALCT Reliability
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(alct_reliability_recipes))
    & (
        (df_cyc.AMSChargeCapacity > alct_reliability_charge_capacity)
        | (df_cyc.dvdt <= alct_reliability_dvdt)
    ),
    "Failed_reliability",
] = True


df_cyc = df_cyc.merge(
    df_cyc[["samplename", "Failed", "Failed_reliability"]].groupby("samplename").max(),
    suffixes=["", "_any"],
    right_index=True,
    left_on="samplename",
)


df_cyc["ShortEvent"] = df_cyc.Failed_any | df_cyc.Failed_reliability


df_cyc_screen = pd.concat(
    [
        df_cyc.loc[
            (df_cyc.ShortEvent == True)
            & ((df_cyc.Failed == True) | (df_cyc.Failed_reliability == True))
        ]
        .groupby("samplename")
        .first(),
        df_cyc.loc[(df_cyc.ShortEvent == False)].groupby("samplename").last(),
    ]
)
df_cyc_screen["EventCycle"] = df_cyc_screen.CumulativeCycle
df_cyc_screen = df_cyc_screen[~df_cyc_screen.index.duplicated()]


df_master = pd.DataFrame(df_cyc["samplename"].unique(), columns=["samplename"]).join(
    df_cyc_screen, on="samplename", how="left"
)


df_master["Build Count"] = 1
df_master["1C Count"] = np.where(
    (
        (
            (df_master.ShortEvent == True)
            & (df_master.idtest_recipe.isin(screen_softstart1C_recipes))
        )
    )
    & (df_master.CumulativeCycle < 50),
    0,
    1,
)

df_master["Fast-Charge Count"] = np.where(
    (
        (
            (
                (df_master.ShortEvent == True)
                & df_master.idtest_recipe.isin(screen_fastcharge)
            )
            | (df_master["1C Count"] == 0)
        )
    )
    & (df_master.CumulativeCycle < 50),
    0,
    1,
)

df_master["C/3 Count"] = np.where(
    (
        ((df_master.ShortEvent == True) & (df_master.idtest_recipe.isin(screen_Co3)))
        | (df_master["Fast-Charge Count"] == 0)
    )
    & (df_master.CumulativeCycle < 30),
    0,
    1,
)

df_master["Yield Count"] = df_master["C/3 Count"]

#df_master["Yield Count"] = 1

df_master["1C Count"] = df_master["1C Count"] * df_master["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(screen_softstart1C_recipes)
    .astype(int)
    .max()
)

df_master["Fast-Charge Count"] = df_master["Fast-Charge Count"] * df_master["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(screen_fastcharge)
    .astype(int)
    .max()
)


df_master["C/3 Count"] = df_master["C/3 Count"] * df_master["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(screen_Co3)
    .astype(int)
    .max()
)

df_master["Reliability Short"] = np.nan

df_master.loc[
    (df_master.ShortEvent == True)
    & (df_master.idtest_recipe.isin(reliability_recipes + alct_reliability_recipes+lowtemp_reliability_recipes))
    & (df_master["Yield Count"] == 1),
    "Reliability Short",
] = True

df_master.loc[
    (df_master.ShortEvent == False)
    & (df_master.idtest_recipe.isin(reliability_recipes + alct_reliability_recipes+lowtemp_reliability_recipes))
    & (df_master["Yield Count"] == 1),
    "Reliability Short",
] = False


df_master.reset_index(inplace=True)

df_master["cell_build_date"] = df_master.groupby("process")[
    "cell_build_time"
].transform("min")
df_master["cell_build_WW"] = (
    df_master["cell_build_date"].dt.isocalendar().year.astype(str)
    + "WW"
    + df_master["cell_build_date"].dt.isocalendar().week.astype(str)
)

df_master["cell_build_date"] = df_master["cell_build_date"].dt.date

df_master["cell flow + date"] = (
    df_master["process"] + ",<br>" + df_master["cell_build_date"].astype(str)
)

df_master["cell_tier_group"] = "Spec Fail"

#%%
#### Ground Truth Tiering of each cell in the ML pouches

# Query dataframe from database
MLgen = genealogy_v2.get_genealogy_2L('APD|ML|UC|QSC', conn) # 
CellsInML = df_master.merge(MLgen, left_on='samplename', right_on='6L_cell_id', how='left')
CellsInML = CellsInML[['samplename', '2L_cell_id']].rename(columns={'2L_cell_id': 'Cell ID'})

print("Pulling Cell Metrics")

## Query HIFI CELL THICKNESS METRICS ##
print("Aquiring Thickness Data")
agent = app.ImageAgent()        
# get US thickness data
df_us_thickness = unit_cell_metro_metrics_ZI.get_thickness_metrics(CellsInML["Cell ID"].str.slice(stop=13).unique(), agent)
# append "_US" to column names
df_us_thickness.columns = [f"{col}_US" for col in df_us_thickness.columns]
df_us_thickness = df_us_thickness.rename(columns={"sample_US": "Cell ID"})
# select desired columns for analysis
selected_columns_df_us_thickness = df_us_thickness[['Cell ID','10mm_eroded_rect_inside_mean_US','0.5mm_eroded_rect_east_mean_US','0.5mm_eroded_rect_west_mean_US','0.5mm_eroded_rect_north_mean_US','0.5mm_eroded_rect_south_mean_US','center_normalized_0.5mm_eroded_rect_outside_mean_US']]
# MERGE US DATA
df_processing = CellsInML.merge(selected_columns_df_us_thickness, how="left", on="Cell ID")
# Define the conditions
conditions = [df_processing['center_normalized_0.5mm_eroded_rect_outside_mean_US'] < 1.1]
# Define the corresponding values
values = [1]
# Assign values to the 'Thickness' column
CellsInML['Thickness'] = np.select(conditions, values, default=3)


## Query Edge Wetting Metrics ##
print("Aquiring Edge Wetting Data")
df_edge_wetting_metrics = unit_cell_metro_metrics_ZI.get_edge_wetting_metrics(CellsInML['Cell ID'].str.slice(stop=13).unique(), agent)
CellsInML = CellsInML.merge(df_edge_wetting_metrics[['sample','median_contour_catholyte_pct']], how="left", left_on="Cell ID", right_on = 'sample').drop(columns = 'sample')

## Query Anode tier for pairing 
print("Aquiring Anode Data")
df_anode_metrics = unit_cell_metro_metrics_ZI.get_anode_tier_A1(CellsInML['Cell ID'].str.slice(stop=13).unique(), agent)
CellsInML['Anode'] = CellsInML['Cell ID'].str.slice(start=0, stop=16).map(df_anode_metrics.set_index('sample')['A1_anode_tier'])
## drop this column or else it will be considered in your ranking and script will bug out 


## Query Cathode Misalignment (Alignment) ##
with ImageClient(host="image-api.qscape.app") as image_client:
#image_client = ImageClient()  # Use ImageClient as a context manager
    print("Aquiring Cathode Misalignment Data")
    for index, row in CellsInML.iterrows():
        sample_name = row['Cell ID']
        image_search = SearchQuery(
            sample_prefix=sample_name,    
            a_type=AnalysisType.CONTRAST,      
            lregex="nordson_matrix-us-stitched-corners$",   
            )
        image_agent = ImageAgent()
        image_results = image_agent.search(query=image_search)
        # Check if 'cathode_alignment_custom_model_prediction' exists in the image results
        if 'cathode_alignment_custom_model_prediction' in image_results:
            cathodemisalignment = image_results['cathode_alignment_custom_model_prediction']
            if cathodemisalignment.iloc[0] == "go":
                CellsInML.at[index, 'Alignment'] = 1
                #print(f"{sample_name} is a 1 based on CV model")
            elif cathodemisalignment.iloc[0] == "no-go":
                CellsInML.at[index, 'Alignment'] = 3
                #print(f"{sample_name} is a 3 based on CV model")

        #Continue with manual review of cathode misalignment and edge wetting
        if pd.notna(sample_name):
            manual_reviews = image_client.get_manual_reviews(samples=[sample_name], include_history=True)
        else:
            print(f"Skipping {row['samplename']} as it has no Unit Cells linked to it")   

        manualreviewCM = convert_manual_reviews_to_dataframe(manual_reviews, include_modified_date=True)

        if 'cathode_alignment' in manualreviewCM:
            if not manualreviewCM.empty and manualreviewCM['cathode_alignment'].notnull().any():
                CellsInML.at[index, 'Alignment'] = manualreviewCM['cathode_alignment'].iloc[0]
                #print(f"Manual Review corrected {sample_name} cathode misalignment to {manualreviewCM['cathode_alignment'].iloc[0]}")

        if 'edge_wetting' in manualreviewCM:
            if not manualreviewCM.empty and manualreviewCM['edge_wetting'].notnull().any():
                CellsInML.at[index, 'median_contour_catholyte_pct'] = manualreviewCM['edge_wetting'].iloc[0]
                #print(f"Manual Review corrected {sample_name} edge wetting to {manualreviewCM['edge_wetting'].iloc[0]}")


conditions = [
    CellsInML['median_contour_catholyte_pct'] < 80,
    (CellsInML['median_contour_catholyte_pct'] >= 80) & (CellsInML['median_contour_catholyte_pct'] <= 98),
    CellsInML['median_contour_catholyte_pct'] > 98
]
choices = [3, 2, 1]
CellsInML['Edge Wetting'] = np.select(conditions, choices)
CellsInML = CellsInML.drop(columns = ['median_contour_catholyte_pct'])

# Calculate the minimum value for each row across the selected columns
CellsInML['Cell Tier'] = np.where(CellsInML['Cell ID'].isna(), np.nan, 
                                  CellsInML[['Thickness', 'Alignment', 'Anode', 'Edge Wetting']].max(axis=1))


# Group by "samplename" and find the max "Cell Tier" for each
FinalMLTier = CellsInML.groupby('samplename', as_index=False)['Cell Tier'].max()
# Rename columns as required
FinalMLTier.columns = ['Multilayer', 'ML Tier']
# Convert "ML Tier" to integer and format as "Tier {max Cell Tier}"
# Conditionally update 'ML Tier'
FinalMLTier['ML Tier'] = np.where(
    FinalMLTier['ML Tier'].isna(), 
    np.nan,  # Keep NaN if it was originally NaN
    "Tier " + FinalMLTier['ML Tier'].fillna(0).astype(int).astype(str)
)

# Merge df_master with FinalMLTier on "samplename" and "Multilayer"
df_master = df_master.merge(FinalMLTier, left_on='samplename', right_on='Multilayer', how='left')

# Update "cell_tier_group" with the values from "ML Tier"
df_master['cell_tier_group'] = df_master['ML Tier']

# Drop the extra "Multilayer" and "ML Tier" columns
df_master = df_master.drop(columns=['Multilayer', 'ML Tier'])








# %%
#Plot Yield Plots
# =============================================================================
# ======================        YIELD PLOTS          ==========================
# =============================================================================



# Group by
grouping = "process"

# grouping = "cell_tier_group"

data = df_master.copy()


# Keep rows where 'cell_tier_group' is 'Tier 1' or 'Tier 2'
data_filtered = data[data['cell_tier_group'].isin(['Tier 1', 'Tier 2'])]



data["cell_build_datetime"] = pd.to_datetime(data["cell_build_date"])
data["date"] = data["cell_build_datetime"].dt.strftime("%a %d %b")

df_cyield = (
    data[
        [
            grouping,
            "cell_build_date",
        ]
    ]  # , "platform"]]  #
    .groupby(grouping)
    .first()
    .join(
        data[
            [
                grouping,
                "Build Count",
                "1C Count",
                "Fast-Charge Count",
                "C/3 Count",
                # "Yield Count",
            ]
        ]
        .groupby(grouping)
        .sum(),
        how="right",
    )
    .reset_index()
).set_index(grouping)


df_cyield[
    [
        "Cells Built",
        "1C Yield",
        "Fast-Charge Yield",
        "C/3 Yield",
    ]
] = 100 * df_cyield[
    [
        "Build Count",
        "1C Count",
        "Fast-Charge Count",
        "C/3 Count",
        # "Yield Count",
    ]
].div(
    df_cyield["Build Count"], axis=0
)


df_cyield = df_cyield.sort_values(grouping)




# Sort values by the grouping column
df_cyield = df_cyield.sort_values(grouping)
df_cyield = df_cyield.sort_values("cell_build_date")


fig = px.bar(
    df_cyield,
    x=df_cyield.index,
    y=[
        "Cells Built",
        "1C Yield",
        "Fast-Charge Yield",
        "C/3 Yield",
        # "Fast-Charge Yield",
        # "Screen Yield",
    ],
    # facet_col="cell_build_date",
    barmode="group",
)



fig.update_xaxes(
    categoryorder="array",
    categoryarray=df_cyield.index.unique(),
)

# Create the text annotations with optional bold formatting
build_count_text = [f"N= {n}" for n in df_cyield["Build Count"]]
# rpt_count_text = [f"<b>N= {n}</b>" for n in df_cyield["C/3 Count"]]
C_3_count_text = [f"<b>N= {n}</b>" for n in df_cyield["C/3 Count"]]
# one_c_count_text = f"<b>N= {df_cyield['1C Count'].values}</b>" 

# Update the traces with the new text lists
fig.data[0].text = build_count_text
# fig.data[1].text = rpt_count_text
fig.data[3].text = C_3_count_text


fig.update_traces(textposition="inside", textfont_size=20)




fig.update_layout(
    xaxis_title=grouping,
    yaxis_title="Screen yield (%)",
    font=dict(
        size=18,
    ),
    legend={"title_text": ""},
    yaxis_range=[0, 100],
)



fig.update_yaxes(tickfont=dict(size=20))
fig.update_xaxes(tickfont=dict(size=18))


fig.update_xaxes(
    categoryorder="array",
#    categoryarray=['APD256AA', 'APD256AB', 'MLB000AB', 'MLB000AC', 'MLB000AD' ]
)


# add grey dotted line at 80% yield
fig.add_shape(
    type="line",
    x0=-0.5,
    x1=df_cyield.shape[0] - 0.5,
    y0=80,
    y1=80,
    line=dict(color="grey", width=2, dash="dot"),
)


# change the bar colors
colors = [
    px.colors.qualitative.Plotly[2],
    px.colors.qualitative.Plotly[3],
    px.colors.qualitative.Plotly[5],
    px.colors.qualitative.Plotly[6],
    px.colors.qualitative.Plotly[4],
    px.colors.qualitative.Plotly[0],
    px.colors.qualitative.Plotly[1],
]
for i in range(len(fig.data)):
    fig.data[i].marker.color = colors[i]


fig.show(renderer="browser")

# %%
#Plot Stacked Bar Chart for Cells Built by Tier


grouped_data = df_master.groupby(["process", "cell_tier_group"])['Build Count'].sum().unstack(fill_value=0)


# Creating a plotly stacked bar chart
fig = go.Figure()

# Add each tier as a separate trace with custom pastel colors
fig.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data['Tier 1'],
    name='Tier 1',
    marker_color=px.colors.qualitative.Pastel1[2]  # Pastel green for Tier 1
))

fig.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data['Tier 2'],
    name='Tier 2',
    marker_color=px.colors.qualitative.Pastel1[1] # Pastel blue for Tier 2
))

fig.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data['Tier 3'],
    name='Tier 3',
    marker_color=px.colors.qualitative.Pastel1[0]  # Pastel red for Tier 3
))

# Update the layout for stacked bar
fig.update_layout(
    barmode='stack',
    title='Total Build Count per Batch, Stacked by Tier',
    xaxis_title='Batch',
    yaxis_title='Total Build Count'
)

fig.update_yaxes(tickfont=dict(size=24), title_font=dict(size=24))
fig.update_xaxes(tickfont=dict(size=24), title_font=dict(size=16))

# add total count label outside each bar

for i, batch in enumerate(grouped_data.index):
    fig.add_annotation(
        x=batch,
        y=grouped_data.loc[batch].sum(),
        text=f"{grouped_data.loc[batch].sum()}",
        showarrow=False,
        font=dict(size=24),
        yshift=10
    )

# increase the font of the legend

fig.update_layout(
    legend=dict(
        title='',
        font=dict(
            size=24
        )
    )   
)



# Show the figure
fig.show(renderer="browser")

# %%

# =============================================================================
# ========================        CELL METRICS          =======================
# =============================================================================

grouping = "process"
color_by = "experiment"

data = df_master.copy()


fig = make_subplots(
    1,
    2,
    horizontal_spacing=0.12,
    vertical_spacing=0.1,
    shared_xaxes=True,
)



# create a color dictionary for each color_by category
color = dict(zip(data[color_by].unique(), px.colors.qualitative.Plotly*5))
color.keys()
data[color_by].unique()

# Set a flag to ensure legend items are added only once
legend_added = {key: False for key in data[color_by].unique()}

for label, group in data[data['C/3 Count']==1].groupby(grouping):
    for color_value, group_color in group.groupby(color_by):
        fig.add_trace(
            go.Box(
                x=group_color[grouping],
                y=group_color["AMSDischargeCapactiy_Co3"],
                quartilemethod="linear",
                name=color_value,
                text=group_color["samplename"],
                showlegend=not legend_added[color_value],
                fillcolor=color[color_value],
                line=dict(color="black"),
            ),
            1,
            1,
        )
        legend_added[color_value] = True


fig.update_yaxes(
    title_text="Discharge Capacity (mAh/g)",
    range=[190, 205],
    row=1,
    col=1,
)

for label, group in data[data['C/3 Count']==1].groupby(grouping):
    for color_value, group_color in group.groupby(color_by):
        fig.add_trace(
            go.Box(
                x=group_color[grouping],
                y=group_color[
                    "DischargeCapactiy_Co3"
                ],  # [group["Final 1C Count"] == 1]
                quartilemethod="linear",
                name=color_value,
                text=group_color["samplename"],
                showlegend=not legend_added[color_value],
                fillcolor=color[color_value],
                line=dict(color="black"),
            ),
            1,
            2,
        )
        legend_added[color_value] = True


fig.update_yaxes(
    title_text="C/3 Discharge Capacity (mAh)",
    range=[5, 7],
    row=1,
    col=2,
)

for i in range(2):
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        row=1,
        col=i + 1,
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        row=1,
        col=i + 1,
    )

fig.update_layout(
    title_text="",
    # xaxis_title=grouping,
    font=dict(
        size=16,
    ),
)

fig.update_traces(boxpoints="all", jitter=0.1)

fig.update_xaxes(
    categoryorder="array",
    categoryarray=data.sort_values(["batch"])[grouping].unique(),
)

fig.show(renderer="browser")


fig = make_subplots(
    1,
    2,
    horizontal_spacing=0.12,
    vertical_spacing=0.1,
    shared_xaxes=True,
)

# plot colors in px.colors.qualitative.Plotly
color = dict(zip(data[color_by].unique(), px.colors.qualitative.Plotly*5))


# Set a flag to ensure legend items are added only once
legend_added = {key: False for key in data[color_by].unique()}

for label, group in data[data['1C Count']==1].groupby(grouping):
    for color_value, group_color in group.groupby(color_by):
        fig.add_trace(
            go.Box(
                x=group_color[grouping],
                y=group_color[
                    "dVdt_1C"
                ],  # [group["Formation Count"] == 1]
                quartilemethod="linear",
                name=color_value,
                text=group_color["samplename"],
                showlegend=not legend_added[color_value],
                fillcolor=color[color_value],
                line=dict(color="black"),
            ),
            1,
            1,
        )
        legend_added[color_value] = True


fig.update_yaxes(
    title_text="dV/dt (µV/s)",
    # range=[20, 30],
    row=1,
    col=1,
)

for label, group in data[data['1C Count']==1].groupby(grouping):
    for color_value, group_color in group.groupby(color_by):
        fig.add_trace(
            go.Box(
                x=group_color[grouping],
                y=group_color["MedDischargeASR_1C"],  # [group["Final 1C Count"] == 1]
                quartilemethod="linear",
                name=color_value,
                text=group_color["samplename"],
                showlegend=not legend_added[color_value],
                fillcolor=color[color_value],
                line=dict(color="black"),
            ),
            1,
            2,
        )
        legend_added[color_value] = True


fig.update_yaxes(
    title_text="1C Discharge ASR (Ohm cm<sup>2</sup>)",
    range=[20, 30],
    row=1,
    col=2,
)

# # add third subplot with ASR_ratio_1C
# for label, group in data.groupby(grouping):
#     for color_value, group_color in group.groupby(color_by):
#         fig.add_trace(
#             go.Box(
#                 x=group_color[grouping],
#                 y=group_color["ASR_ratio_1C"],  # [group["Final 1C Count"] == 1]
#                 quartilemethod="linear",
#                 name=color_value,
#                 text=group_color["samplename"],
#                 showlegend=False,
#                 fillcolor=color[color_value],
#                 line=dict(color="black"),
#             ),
#             1,
#             3,
#         )

# fig.update_yaxes(
#     title_text="Charge/Discharge ASR Ratio",
#     range=[0.8, 1.2],
#     row=1,
#     col=3,
# )

for i in range(2):
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        row=1,
        col=i + 1,
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        row=1,
        col=i + 1,
    )


fig.update_layout(
    # xaxis_title=grouping,
    font=dict(
        size=16,
    ),
    # show legend
    showlegend=True,
    height=700,
    width=2000,
)

fig.update_traces(boxpoints="all", jitter=0.1)

fig.update_xaxes(
    categoryorder="array",
    categoryarray=data.sort_values(["batch"])[grouping].unique(),
)  #

fig.show(renderer="browser")



#%%# %%
# Query and Plot Reliability
# ====================================================================================
# ======================        Reliability        =====================================
# ====================================================================================


# define function for meging dataframes
def merge_on_common_cols(df1: pd.DataFrame, df2: pd.DataFrame):
    """merge on cols in common between two dfs"""
    common_cols = [c for c in df1.columns if c in df2.columns]
    print(f"merging on common cols: {common_cols}")
    return df1.merge(df2, on=common_cols).copy()

# set up input variables
sample_regix = search
track_cycle_dvdt_cutoff = -1.5E-5

# split sample regex into list
sample_prefixes = sample_regix.split("|")

test_type = "E31"
recipe_ids = [15223, 15224, 15263, 15416, 15410, 15287, 15445, 15411, 15529, 15551, 15707 ]

## Fetch Data
# get meta data about runs for a specific sample prefix and test type
run_info_df = qs_client.get_run_info(sample_prefixes=sample_prefixes, test_type=test_type, recipe_ids=recipe_ids)
print(f"{len(run_info_df)} runs found")

run_info_df.loc[:, "run_id"] = run_info_df["run_id"].apply(lambda r: int(r))
print(f"{len(run_info_df)} valid runs found")
run_ids = [int(x) for x in run_info_df['run_id']]

# add new columns, miscellaneous
run_info_df.loc[:, "Process Name"] = run_info_df["batch_name"].apply(lambda s: s[:8])

# drop null columns
run_info_df = run_info_df.dropna(axis=0, how="all")

# get the cycle metrics for every run id in run_ids
# this step can take many minutes
cycle_metrics_df = qs_client.get_et_cycle_metrics(run_ids=run_ids, test_type=test_type)

# drop columns will null values
cycle_metrics_df.dropna(subset=["voltage_post_ceiling_rest_end_linear_dvdt", "min_track_cycle_power", "voltage_end_floor_rest", "min_track_cycle_voltage"], inplace=True)

# add column to flag shorted cycles
cycle_metrics_df.loc[:, "is_shorted"] = cycle_metrics_df["voltage_post_ceiling_rest_end_linear_dvdt"].apply(lambda dvdt: dvdt < track_cycle_dvdt_cutoff)
cycle_metrics_df.loc[:, "V_fail_2.45V"] = cycle_metrics_df["min_track_cycle_voltage"].apply(lambda vmin: vmin < 2.45)

# add column to calculate max overpotential
cycle_metrics_df.loc[:, "max_overpotential_V"] = cycle_metrics_df.apply(lambda row: row["voltage_end_floor_rest"] - row["min_track_cycle_voltage"], axis=1)

# merge cycle metrics and run info

merged_e31_cycle_metrics_df = merge_on_common_cols(cycle_metrics_df, run_info_df)

all_e31_cycle_metrics_df = merged_e31_cycle_metrics_df.copy()


## Plot Vmin vs Cycle Count
plotly_colors=[    
            'rgb(255, 127, 14)',
            'rgb(44, 160, 44)', 
            'rgb(214, 39, 40)',
            'rgb(31, 119, 180)', 
            'rgb(148, 103, 189)', 
            'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 
            'rgb(127, 127, 127)',
            'rgb(188, 189, 34)',
            'rgb(23, 190, 207)']*10

color_list=plotly_colors

df_Vmin_plot=all_e31_cycle_metrics_df[['sample_name', "Process Name", "track_cycle_count_cumulative","min_track_cycle_voltage" ]]

df_Vmin_plot['experiment'] = df_Vmin_plot['sample_name'].str[0:6]
# Sort the DataFrame by 'sample_name'
#df_Vmin_plot = df_Vmin_plot.sort_values('sample_name')

group_by_col='experiment'

#df_Vmin_plot = df_Vmin_plot[df_Vmin_plot['sample_name'].str.contains('MLB|MLD')]

color_dict = {}
for group in df_Vmin_plot.groupby(group_by_col):
    if group[0] not in color_dict.keys():
        color_dict[group[0]] = {}
    color_dict[group[0]] = (color_list)[len(color_dict.keys())-1]


# create dictionary mapping samplenames to colors with the same color for each sample in group_by_col
sample_color_dict = {}
for group in df_Vmin_plot.groupby(group_by_col):
    for sample in group[1]['sample_name'].unique():
        if sample not in sample_color_dict.keys():
            sample_color_dict[sample] = {}
        sample_color_dict[sample] = color_dict[group[0]]

# plot min vs count in plotly 

fig = px.scatter(df_Vmin_plot, x="track_cycle_count_cumulative", y="min_track_cycle_voltage", color='sample_name', title='Min Discharge Voltage vs Track Cycle Count', hover_name='sample_name',
                 color_discrete_map=sample_color_dict
)

# Sort the traces in alphabetical order by 'sample_name'
sorted_sample_names = sorted(df_Vmin_plot['sample_name'].unique())  # Get unique sample names and sort them alphabetically
fig.for_each_trace(lambda trace: trace.update(legendgroup=sorted_sample_names.index(trace.name),
                                              legendrank=sorted_sample_names.index(trace.name)))

fig.update_yaxes(range=[2.2, 3.0], title='Voltage (V)', tickfont=dict(size=22), 
                # dtick=0.5,
                    titlefont=dict(size=22), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')    

fig.update_xaxes(range=[0, 60], title='Cycle Number', tickfont=dict(size=22), 
                # dtick=0.5,
                    titlefont=dict(size=22), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')  



# add markers and lines
fig.update_traces(mode='markers+lines', marker=dict(size=10), line=dict(width=2))


fig.update_layout(
    autosize=False,
    width=1200,
    height=700,
    font = dict(size = 20),
    plot_bgcolor='white',
)

# add dotted line at 2.0V
fig.add_shape(type="line",
    x0=-5, y0=2.45, x1=60, y1=2.45,
    line=dict(color="black",width=2, dash="dot")
)
#fig.show()
fig.show(renderer="browser")


## Plot Track Cycle Reliability
from lifelines import KaplanMeierFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
from scipy.stats import weibull_min, norm


grouping = "experiment"
df_rel_master_voltage = all_e31_cycle_metrics_df.copy()
df_rel_master_voltage['samplename'] = df_rel_master_voltage['sample_name']

Vmin_cut = 'V_fail_2.45V'
Vmin_cut_any = 'V_fail_2.45V'

df_rel_master_voltage = df_rel_master_voltage[df_rel_master_voltage['sample_name'].str.contains('MLB|MLD')]

#df_rel_master_voltage.loc[df_rel_master_voltage.samplename.str.contains('APD256'), 'Condition'] = 'A1 6L'
#df_rel_master_voltage.loc[df_rel_master_voltage.samplename.str.contains('APD253|QSC020'), 'Condition'] = 'A1 22L'

df_rel_master_voltage['experiment'] = df_rel_master_voltage['samplename'].str[0:6]
df_rel_master_voltage['process'] = df_rel_master_voltage['samplename'].str[0:8]
df_rel_master_voltage['batch'] = df_rel_master_voltage['samplename'].str[0:13]


df_Vmin = df_rel_master_voltage.merge(
    df_rel_master_voltage[["samplename", Vmin_cut]].groupby("samplename").max(),
    suffixes=["", "_any"],
    right_index=True,
    left_on="samplename",
)

v_fail_screen=pd.concat([
    df_Vmin.loc[(df_Vmin[Vmin_cut]==True) & (df_Vmin[Vmin_cut_any]==True) ].groupby('samplename').first(),
    df_Vmin.loc[(df_Vmin[Vmin_cut]==False) & (df_Vmin[Vmin_cut_any]==False)].groupby('samplename').last()]
)


v_fail_screen=v_fail_screen[['track_cycle_count_cumulative', Vmin_cut]].reset_index()
v_fail_screen=v_fail_screen.rename(columns={'track_cycle_count_cumulative': 'V_Fail_Cycle'})

df_rel_master_voltage = pd.merge(df_rel_master_voltage, v_fail_screen, on=['samplename',Vmin_cut], how='left')

annotate = True
six_layer = False
RMST_duration=40

charge_cap_cutoff=210

width=1200
height=800
range_x=100


df_rel_master_voltage = df_rel_master_voltage.merge(
    df_rel_master_voltage[["samplename", "is_shorted"]].groupby("samplename").max(),
    suffixes=["", "_any"],
    right_index=True,
    left_on="samplename",
)

df_rel_master_voltage['Fail_Event'] = False

df_rel_master_voltage["ShortEvent"] = df_rel_master_voltage['is_shorted_any']
df_rel_master_voltage['EventCycle'] = df_rel_master_voltage['track_cycle_count_cumulative']

#label builds that survived
df_rel_master_voltage.loc[(df_rel_master_voltage['is_shorted_any'] == False), 'Failure_Type' ]='Survived'

#label builds that failed via shorting
df_rel_master_voltage.loc[(df_rel_master_voltage['is_shorted_any'] == True), 'Fail_Event' ]=True
df_rel_master_voltage.loc[(df_rel_master_voltage['is_shorted_any'] == True), 'Failure_Type' ]='Short'

# label builds that failed via Vmin 
df_rel_master_voltage.loc[((df_rel_master_voltage[Vmin_cut]==True)) , "Fail_Event"]=True
df_rel_master_voltage.loc[((df_rel_master_voltage[Vmin_cut]==True) & (df_rel_master_voltage['V_Fail_Cycle'] < df_rel_master_voltage['EventCycle'])) , ["Fail_Event", "Failure_Type" ]]=[True, 'Vmin Failure']
df_rel_master_voltage.loc[df_rel_master_voltage[Vmin_cut]==True, 'EventCycle'] = df_rel_master_voltage['V_Fail_Cycle']


df_rel_master_voltage_summary=df_rel_master_voltage[['samplename', 'EventCycle', 'Fail_Event', 'Failure_Type','run_end_time', 'recipe_id', 'recipe_name', 'tool_name', 'channel', 'charge_capacity_cumulative', grouping]].copy()
# df_rel_master_voltage_summary.loc[df_rel_master_voltage_summary.samplename.str.contains('APD256AA'), ['Fail_Event', 'Failure_Type']]=[False,'Survived']
df_rel_master_voltage_summary = df_rel_master_voltage_summary.drop_duplicates(['samplename'], keep = 'last')

## make survival plot
fig = make_subplots()
fill_color_list=['rgba'+ a[3:-1]+', 0.06)' for a in color_list]
i=0

kmf1 = KaplanMeierFitter(alpha=0.05)  # this alpha is the Type I error rate

from lifelines.utils import restricted_mean_survival_time
results_df = pd.DataFrame(columns=['Condition', 'RMST', 'Variance', '95_CI'])

#df_combined_forcsv.to_csv('dvdt_EventCycle_240222.csv')

for Batch, grouped in df_rel_master_voltage_summary.groupby(grouping):
    
    kmf1.fit(durations=grouped["EventCycle"], event_observed=grouped["Fail_Event"])
    df = kmf1.survival_function_.join(kmf1.confidence_interval_survival_function_)
    df = df.join(
        grouped.set_index("samplename")[["EventCycle", "Fail_Event", "Failure_Type"]]
        .reset_index()
        .groupby(["EventCycle", "Fail_Event", "Failure_Type"])
        .agg({"samplename": "<br>\n".join})
        .reset_index()
        .set_index("EventCycle")
    )



    df = df.fillna(value=True)
    df["Fail_Event"] = df["Fail_Event"].apply(int)
    df.loc[df.index==0, "Fail_Event" ] = 0
    df['color']=color_list[i]

    df['Fail_Short']=0
    df.loc[df['Failure_Type']=='Short', 'Fail_Short']=1
    df.loc[df['Failure_Type']=='Voltage', 'color']='yellow'

    # Calculate RMST and variance
    rmst, variance = restricted_mean_survival_time(kmf1, t=RMST_duration, return_variance=True)
    
    standard_error = np.sqrt(variance)
    z_score = 1.96  # for 95% confidence interval

    # Compute confidence intervals
    ci = z_score * standard_error
    

    # Append results to the DataFrame
    results_df = results_df.append({'Condition': Batch,
                                    'RMST': rmst,
                                    'Variance': variance,
                                    '95_CI': ci}, ignore_index=True)
    

    if six_layer:
        if '6L' not in Batch:
            df['KM_estimate']=df['KM_estimate']**3
            df['KM_estimate_lower_0.95']=df['KM_estimate_lower_0.95']**3
            df['KM_estimate_upper_0.95']=df['KM_estimate_upper_0.95']**3


    trace1 = {
        "x": df.index,
        "y": df.KM_estimate,
        "line": {"shape": "hv"},
        "mode": "lines",
        "name": "value",
        "type": "scatter",
    }

    df=df.dropna(subset='Fail_Event')

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.KM_estimate * 100,
            mode="markers+lines",
            line=dict(shape="hv", width=3, color=color_list[i]),
            marker=dict(color=df['color'], symbol='circle', size=7*(1-df['Fail_Short'])),
            hovertext=df.samplename,
            name=f"{Batch} (N={len(grouped)})",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["KM_estimate_upper_0.95"] * 100,
            mode="lines",
            line=dict(shape="hv", width=0, color=color_list[i]),
            name="",  # f"{Batch} UCI95%",
            showlegend=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["KM_estimate_lower_0.95"] * 100,
            mode="lines",
            fill="tonexty",
            fillcolor=fill_color_list[i],
            line=dict(shape="hv", width=0, color=color_list[i]),
            name="",  # f"{Batch} LCI95%",
            showlegend=False,
        ),
        secondary_y=False,
    )
    i+=1



fig.update_layout(
    title="Reliability Test",
    xaxis=dict(title="Cycle Number"),
    yaxis=dict(title="Survival (%)"),
    font=dict(size=20),
    legend={"traceorder": "normal"},
    legend_title_text=grouping,
    # autosize=False,
    width=1050,
    height=600,
    # hide the legend
    # showlegend=False,
)

# set background color to white
fig.update_layout(plot_bgcolor='white')
fig.update_yaxes(range=[0, 105], showline=True, linewidth=1, linecolor="black", mirror=True)
fig.update_xaxes(range=[0, 100], showline=True, linewidth=1, linecolor="black", mirror=True)


if annotate:
# add vertical grey dashed line to figure at 60 cycles
    fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=60,
                y0=0,
                x1=60,
                y1=105,
                line=dict(
                    color="Grey",
                    width=3,
                    dash="dash",
                ),
            )
        )

# add red circle at 60 cycles and 50% survival

    # fig.add_trace(
    #     go.Scatter(
    #         x=[60],
    #         y=[88],
    #         mode="markers",
    #         marker=dict(color="red", symbol="circle", size=20),
    #         hovertext='50% Survival',
    #         name="",
    #         # remove from legend
    #         showlegend=False,
    #     ),
    #     secondary_y=False,
    # )

fig.update_yaxes(range=[0, 105])
fig.update_xaxes(range=[0, range_x])

fig.show(renderer="browser")
#fig.show()

# %%
# Produce ML Summary Spreadsheet
# ====================================================================================
# ====================  Yield and Reliability Summary   ==============================
# ====================================================================================

#Step 1: Create dataframes that summarizes ML Screen and Reliability Performances
#bring up summary of screen data
df_screening = df_master[['samplename', 'cell_tier_group', 'Yield Count', '1C Count', 'Fast-Charge Count', 'C/3 Count', 'cell_build_date', 'Tool', 'Channel']]
df_screening.rename(columns={"Yield Count": "ML Screen"}, inplace=True)
df_screening['ML Screen'] = df_screening['ML Screen'].replace({1: 'Pass', 0: 'Fail'})
df_failed = df_master[(df_master["C/3 Count"] == 0)][['samplename', "1C Count", "Fast-Charge Count", 'C/3 Count', 'cell_tier_group', 'cell_build_date', 'Tool', 'Channel']]
df_passed = df_master[(df_master["C/3 Count"] == 1)][['samplename', "1C Count", "Fast-Charge Count", 'C/3 Count', 'cell_tier_group', 'cell_build_date', 'Tool', 'Channel']]
#bring up summary of reliability data
rel_summary = df_rel_master_voltage_summary.copy()

# Step 2: Merge screen and reliability dataframes
df_screening = df_screening.merge(rel_summary, on='samplename', how='left')
df_screening['Fail_Event'] = df_screening['Fail_Event'].replace({False: 'Pass', True: 'Failed'})
df_screening = df_screening.rename(columns={'Fail_Event': 'Reliability Result'})
df_screening = df_screening.rename(columns={'EventCycle': 'Total Reliability Cycles'})
df_screening = df_screening.rename(columns={"run_end_time": "Last Reliability Cycle"})
df_screening['Last Reliability Cycle'] = df_screening['Last Reliability Cycle'].str[:10]
df_screening = df_screening[['samplename','cell_tier_group', 'cell_build_date', 'Tool', 'Channel', 'ML Screen', 'Reliability Result', 'Total Reliability Cycles','Last Reliability Cycle', 'recipe_name']]

# Step 3: Update "Reliability Result" based on conditions
today = datetime.today().date()
df_screening['Last Reliability Cycle'] = pd.to_datetime(df_screening['Last Reliability Cycle'], errors='coerce')

df_screening['Reliability Result'] = np.where(
    (df_screening['Reliability Result'] == 'Pass') & 
    ((df_screening['Last Reliability Cycle'].dt.date == today) | df_screening['Last Reliability Cycle'].isna()),
    'In-Progress',
    np.where(
        (df_screening['Reliability Result'] == 'Pass') & 
        (df_screening['Last Reliability Cycle'].dt.date != today),
        'Stopped/Finished',
        df_screening['Reliability Result']
    )
)


# Step 4: Update Maccor and Channel if cell is in reliability testing
df_updated = df_screening.merge(
    df_rel_master_voltage_summary[['samplename', 'tool_name', 'channel']],
    on='samplename',
    how='left',
    suffixes=('', '_new')
)
df_updated['Tool'] = df_updated['tool_name'].combine_first(df_updated['Tool'])
df_updated['Channel'] = df_updated['channel'].combine_first(df_updated['Channel'])
df_updated = df_updated.drop(columns=['tool_name', 'channel'])

# Step 5: Sort by 'cell_build_date' first, and then by 'samplename'
df_screening = df_updated.sort_values(by=['cell_build_date', 'samplename'], ascending=[True, True])

# Step 6: Display the updated dataframe
df_screening.to_clipboard(index=False)

df_screening['Current Status']='0'

# %%