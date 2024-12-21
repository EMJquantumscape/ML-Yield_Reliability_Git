# %%
import pandas as pd
import numpy as np
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
import anode_metrics_hifiscds as am
import genealogy
import mass

# import unit_cell_electrical_yield_and_metrics as uceym
import unit_cell_electrical_yield_and_metrics_with_rel as uceym_rel

# create the quantumscape data client
qs_client = Client()

# %%

# This is the experiment code we want to look at. Default:
search = "MLD018|MLD012|MLD016"
exclude = "None"


# Yield criteria

screen_softstart1C_recipes = [13753, 14399, 14419, 14440, 15490]
softstart1C_charge_capacity_fraction = 0.999
softstart1C_dvdt = -8.0
softstart1C_delta_dvdt = 1
softstart1C_CE = 0.98
softstart1C_ceiling_hold_time = 3600

# screen_fastcharge = [13775, 14445]
# fastcharge_charge_capacity_fraction = 0.95
# fastcharge_dvdt = -10
# fastcharge_delta_dvdt = 2
# fastcharge_CE = 0.98
# fastcharge_ceiling_hold_time = 3600

screen_Co3 = [13708, 15618]  # , 13213,13197, 13708, 13345, ]
Co3_charge_capacity = 202
Co3_dvdt = -10
Co3_charge_capacity_fraction = 1.1
Co3_charge_capacity_fraction_cycle = 1.1

screen_Co3_RPT = [15422]  # , 13213,13197, 13708, 13345, ]
Co3_charge_capacity = 202
Co3_dvdt = -10


# Query data
conn = qs_client.get_mysql_engine()

recipes = "|".join(
    [
        str(x)
        for x in (
            screen_Co3
            + screen_Co3_RPT
            + screen_softstart1C_recipes
            # + screen_fastcharge
           
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


df_cyc = df_cyc.set_index("samplename")


df_cyc["AMSDischargeCapactiy_1C"] = (
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["AMSDischargeCapacity"]
    ]
    .groupby("samplename")
    .min()
)


df_cyc["AMSDischargeCapactiy_Co3_first"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["AMSDischargeCapacity"]]
    .groupby("samplename")
    .first()
)

df_cyc["AMSDischargeCapactiy_Co3_last"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["AMSDischargeCapacity"]]
    .groupby("samplename")
    .last()
)


df_cyc["DischargeCapactiy_Co3_first"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeCapacity"]]
    .groupby("samplename")
    .first()
)

df_cyc["DischargeCapactiy_Co3_last"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeCapacity"]]
    .groupby("samplename")
    .last()
)

df_cyc["DischargeEnergy_Co3_first"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeEnergy"]]
    .groupby("samplename")
    .first()
)

df_cyc["DischargeEnergy_Co3_last"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeEnergy"]]
    .groupby("samplename")
    .last()
)

df_cyc["AMSChargeCapacity_Co3_RPT"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3_RPT), ["AMSChargeCapacity"]]
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


df_cyc["ASR_delta_1C"] = np.abs(
    df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        [],
    ]
    .groupby("samplename")
    .max("MedDischargeASR")
    - df_cyc.loc[
        df_cyc.idtest_recipe.isin(screen_softstart1C_recipes)
        & (df_cyc["CycleIndex"] > 1),
        ["MedDischargeASR"],
    ]
    .groupby("samplename")
    .min()
)




df_cyc["dVdt_Co3"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["dvdt"]]
    .groupby("samplename")
    .last()
)

df_cyc["dVdt_1C_min"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["dvdt"]]
    .groupby("samplename")
    .min()
)

df_cyc["CE_1C_min"] = (
    df_cyc.loc[df_cyc.idtest_recipe.isin(screen_softstart1C_recipes), ["CE"]]
    .groupby("samplename")
    .min()
)



df_cyc["Charge_Fraction_Co3_first"] = (
    df_cyc["AMSChargeCapacity_Co3_RPT"]/df_cyc["AMSDischargeCapactiy_Co3_first"]
)

df_cyc["Charge_Fraction_Co3_last"] = (
    df_cyc["AMSChargeCapacity_Co3_RPT"]/df_cyc["AMSDischargeCapactiy_Co3_last"]
)



df_cyc = df_cyc.reset_index()

# If it stopped on short, it failed
df_cyc["Failed"] = df_cyc["StoppedOnShort"] == 1
df_cyc["Failed_reliability"] = df_cyc["StoppedOnShort"] == 1



# C/3 cycle
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(screen_Co3+screen_Co3_RPT))
    & (
        (
            (df_cyc.ChargeCapacityFraction  > Co3_charge_capacity_fraction )
            # | (df_cyc.AMSChargeCapacity < 100)
        )
        | (df_cyc.dvdt <= Co3_dvdt)
    ),
    "Failed",
] = True

# 1C cycle
df_cyc.loc[
    (df_cyc.idtest_recipe.isin(screen_softstart1C_recipes))
    & (
        (
            (df_cyc.ChargeCapacityFraction  > softstart1C_charge_capacity_fraction)
            | (df_cyc.dvdt<softstart1C_dvdt)
            | (df_cyc.dVdt_delta_1C > softstart1C_delta_dvdt)
        )
    ),
    "Failed",
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


# df_master.loc[((df_master.Charge_Fraction_Co3_first>Co3_charge_capacity_fraction_cycle) | (df_master.Charge_Fraction_Co3_last>Co3_charge_capacity_fraction_cycle)), "ShortEvent"] = True

df_master["Build Count"] = 1


df_master["C/3 Count"] = np.where(
    (
        ((df_master.ShortEvent == True) & (df_master.idtest_recipe.isin(screen_Co3+screen_Co3_RPT)))
    )
    & (df_master.CumulativeCycle < 30),
    0,
    1,
)

df_master["1C Count"] = np.where(
    (
        (
            (df_master.ShortEvent == True)
            & (df_master.idtest_recipe.isin(screen_softstart1C_recipes))
        )
        | (df_master["C/3 Count"] == 0)
    )
    & (df_master.CumulativeCycle < 20),
    0,
    1,
)

df_master["Yield Count"] = df_master["1C Count"]


# df_master["Reliability Short"] = np.nan

# df_master.loc[
#     (df_master.ShortEvent == True)
#     & (df_master.idtest_recipe.isin(reliability_recipes + alct_reliability_recipes))
#     & (df_master["Yield Count"] == 1),
#     "Reliability Short",
# ] = True

# df_master.loc[
#     (df_master.ShortEvent == False)
#     & (df_master.idtest_recipe.isin(reliability_recipes + alct_reliability_recipes))
#     & (df_master["Yield Count"] == 1),
#     "Reliability Short",
# ] = False


df_master.reset_index(inplace=True)

df_master["C/3 Count"] = df_master["C/3 Count"] * df_master["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(screen_Co3)
    .astype(int)
    .max()
)

df_master["1C Count"] = df_master["1C Count"] * df_master["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(screen_softstart1C_recipes)
    .astype(int)
    .max()
)


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



# ====================== ASSIGN CELL TIER GROUP ==========================

df_master["cell_tier_group"] = "Spec Fail"

# if PSOO, then Cell Tier = 1a, if PS01 then Cell Tier = 1b, if PS02 then Cell Tier = Spec Fail
df_master.loc[
    df_master["samplename"].str.contains('MLD018'),
    "cell_tier_group",
] = "Tier 1"

df_master.loc[
    df_master["samplename"].str.contains('MLD016'),
    "cell_tier_group",
] = "Tier 2"

df_master.loc[
    df_master["samplename"].str.contains('MLD012'),
    "cell_tier_group",
] = "Tier 3"


# copy to clipboard
df_master[
    [
        "samplename",
        "cell_build_date",
        "cell_tier_group",
        "Yield Count",
        "AMSDischargeCapactiy_Co3_first",
        "AMSDischargeCapactiy_Co3_last",
        "DischargeCapactiy_Co3_first",
        "DischargeCapactiy_Co3_last",
        "DischargeEnergy_Co3_first",
        "DischargeEnergy_Co3_last",
        "dVdt_Co3",
        "Tool",
        "Channel",
    ]
].to_clipboard(index=False)

# %%
# =============================================================================
# ======================        YIELD PLOTS          ==========================
# =============================================================================



# Group by
grouping = "date"

# grouping = "cell_tier_group"

data = df_master.copy()

# data = data[data.cell_tier_group.str.contains('Tier 1a|Tier 1b')]

data = data[data.samplename.str.contains('MLD018|MLD016')]

data["cell_build_datetime"] = pd.to_datetime(data["cell_build_date"])
data['date'] = data["cell_build_datetime"].dt.strftime('%m/%d/%Y')


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
                "C/3 Count",
                "1C Count",
                # "Fast-Charge Count",
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
        "RPT Yield",
        "1C Yield",
        # "Fast-Charge Yield",
        # "Screen Yield",
    ]
] = 100 * df_cyield[
    [
        "Build Count",
        "C/3 Count",
        "1C Count",
        # "Fast-Charge Count",
        # "Yield Count",
    ]
].div(
    df_cyield["Build Count"], axis=0
)


df_cyield = df_cyield.sort_values(grouping)



fig = px.bar(
    df_cyield,
    x=df_cyield.index,
    y=[
        "Cells Built",
        "RPT Yield",
        "1C Yield",
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
rpt_count_text = [f"<b>N= {n}</b>" for n in df_cyield["C/3 Count"]]
one_c_count_text = [f"<b>N= {n}</b>" for n in df_cyield["1C Count"]]
# one_c_count_text = f"<b>N= {df_cyield['1C Count'].values}</b>" 

# Update the traces with the new text lists
fig.data[0].text = build_count_text
fig.data[1].text = rpt_count_text
fig.data[2].text = one_c_count_text


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
    categoryarray=['QSC020AA', 'QSC020AB', 'QSC020AC', 'QSC020AD', 
    # 'Reused<br>Catholyte',  'Tier 2', 
    'QSC020AE', 'QSC020AF',
       'QSC020AG', 'QSC020AJ', 'QSC020AK',
     'QSC020AL', 'QSC020AM', 'QSC020AN',
       'QSC020AP' ]
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



# print("Tier1a Cells Built: ", df_cyield.loc["Tier 1a", "Build Count"])
# print("Tier1b Cells Built: ", df_cyield.loc["Tier 1b", "Build Count"])
# # print("Tier2 Cells Built: ", df_cyield2.loc["Tier 2", "Build Count"])

# print("Tier1a Cells Yielded: ", df_cyield.loc["Tier 1a", "C/3 Count"])
# print("Tier1b Cells Yielded: ", df_cyield.loc["Tier 1b", "C/3 Count"])
# # print("Tier2 Cells Yielded: ", df_cyield2.loc["Tier 2", "C/3 Count"])

# print("Total Cells Built: ", df_cyield["Build Count"].sum())
# print("Total Cells Yielded: ", df_cyield["C/3 Count"].sum())

# %%

# df_master_summary = df_master[(df_master['1C Count']==1)]

df_master_summary = df_master.copy()

# df_master_summary = df_master_summary[['samplename', 'AMSDischargeCapactiy_1C', 'AMSDischargeCapactiy_Co3_first', 'AMSDischargeCapactiy_Co3_last',
#                                  'DischargeEnergy_Co3_first', 'DischargeEnergy_Co3_last', 'AMSChargeCapacity_Co3_RPT', 'MedDischargeASR_1C', 'dVdt_delta_1C',
#                                    'cell_build_date', 'cell_build_WW', 'cell_tier_group']]


df_master_summary = df_master_summary[['samplename', "Charge_Fraction_Co3_first", 'dVdt_1C_min', "CeilingHoldTime", "CE",
                                  'MedDischargeASR_1C',  'cell_build_date', 'cell_build_WW', 'cell_tier_group', "C/3 Count", '1C Count']]


df_master_summary.to_csv("df_master_summary.csv", index=False)


# %%
# ===========================================================================================================
# ======================        CELLS BUILT/YIELDED BY DATE/TIER        =====================================
# ===========================================================================================================



#Plot Stacked Bar Chart for Cells Built by Tier

df_master2 = df_master[df_master['samplename'].str.contains('MLD018')]



df_master2["cell_build_datetime"] = pd.to_datetime(df_master2["cell_build_date"])
df_master2['date'] = df_master2["cell_build_datetime"].dt.strftime('%m/%d/%Y')


df_master2 = df_master2.sort_values("cell_build_date")

grouped_data = df_master2.groupby(["date", "cell_tier_group"])['Build Count'].sum().unstack(fill_value=0)


# Creating a plotly stacked bar chart
fig = go.Figure()

# Add each tier as a separate trace with custom pastel colors
fig.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data['Tier 1'],
    name='Tier 1',
    marker_color=px.colors.qualitative.Pastel1[2]  # Pastel green for Tier 1
))

# check if Tier 2 and Tier 3 exist in the grouped data
if 'Tier 2' in grouped_data.columns:
    fig.add_trace(go.Bar(
        x=grouped_data.index,
        y=grouped_data['Tier 2'],
        name='Tier 2',
        marker_color=px.colors.qualitative.Pastel1[1]  # Pastel blue for Tier 2
    ))

if 'Tier 3' in grouped_data.columns:
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

fig.update_yaxes(range=[0,8], tickfont=dict(size=24), title_font=dict(size=24))
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
# ========================= Pull RPT AST Data  ================================
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from et.cloud import Cloud
import qs.et.etp_pb2 as etp_pb2
import qs.et.types_pb2 as types_pb2
from qs.et.v2.filters_pb2 import FilterSet, EqualsFilter
from qs.et.v2.search_pb2 import RunSearchRequest

from qsdc.client import Client

# create our data client
qs_client = Client()

cld = Cloud()

samples = df_master["samplename"].unique()

# file_prefix = "/mnt/c/Users/JAM04/OneDrive - Quantumscape Corporation/JMP"
run_info_df = qs_client.get_run_info(sample_prefixes=samples, test_type="E10")
run_id_list = list(run_info_df["run_id"])




def get_e10_runs_on_sample(cld: Cloud, sample_name: str) -> list[int]:
    """Finds all E10 runs on a given sample."""
    search_filters = FilterSet(
        equals = [
            EqualsFilter(field_name="test_type", string_value="E10"),
            EqualsFilter(field_name="sample_name", string_value=sample_name)
        ]
    )
    search_request = RunSearchRequest(filters=search_filters)
    search_response = cld.search_runs(search_request)
    return [v.run_id for v in search_response.runs]

def get_e10_runs_on_samples(cld: Cloud, sample_names: list) -> list[int]:
    """inefficiently loops through samples -> TODO group by prefix instead"""
    if len(sample_names) > 0:
        #print(get_e10_runs_on_sample(cld, sample_names[0]))
        #print(sample_names[1:])
        return get_e10_runs_on_sample(cld, sample_names[0]) + get_e10_runs_on_samples(cld, sample_names[1:])
    else:
        return []

def get_step_metrics(cld: Cloud, run_id: int) -> pd.DataFrame:
    """Fetch the step metrics for the given Run ID."""
    metrics_request = etp_pb2.MetricsQueryRequest()
    metrics_request.query.level = types_pb2.ResolutionLevel.STEP
    metrics_request.run_list.runs.append(run_id)
    metrics_response = cld.fetch_processed(metrics_request)
    return metrics_response.df_steps


def get_pulse_metrics(cld: Cloud, run_id: int) -> pd.DataFrame:
    """Fetch the pulse metrics for the given Run ID."""
    metrics_request = etp_pb2.MetricsQueryRequest()
    metrics_request.query.level = types_pb2.ResolutionLevel.PULSE
    metrics_request.run_list.runs.append(run_id)
    metrics_response = cld.fetch_processed(metrics_request)
    run_df = metrics_response.df_pulses
    run_df.loc[:, "run_id"] = int(run_id)
    return run_df


def get_pulse_metrics_on_runs(cld: Cloud, run_ids: list) -> pd.DataFrame:
    """Return a df of pulse metrics from multiple runs"""
    return pd.concat([get_pulse_metrics(cld, run_id) for run_id in run_ids])#, axis=1)

# retrieve pulse metrics (many minutes)

pulse_metrics = get_pulse_metrics_on_runs(cld, [int(r) for r in run_id_list])
print(f"{len(pulse_metrics)} found")

pulse_metrics = pulse_metrics.dropna(subset=["DCIR_charge_resistance_set_01"]).copy()

run_info_df.loc[:, "run_id"] = run_info_df["run_id"].apply(lambda s: int(s))  # first match run_id types
merged_pulse_df = pd.merge(pulse_metrics, run_info_df, on=["run_id"])


def reshape_E10_pulse_metrics(pulse_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in pulse_df.columns if "set" not in c]
    set_cols = [c for c in pulse_df.columns if "set" in c]
    set_prefixes = pd.Series(["_".join(c.split("_")[0:-2]) for c in set_cols]).unique()
    sets = pd.Series([c.split("_")[-1] for c in set_cols]).unique()
    #print(keep_cols)
    #print(set_cols)
    #print(set_prefixes)
    #print(sets)

    per_set_cols = np.append(["set"], set_prefixes)
    #print(per_set_cols)
    reshaped_df = pd.DataFrame()

    for ind, row in pulse_df.iterrows():
        set_data_dict = {c: [] for c in per_set_cols}
        # define and fill out new columns
        for set in sets:
            set_data_dict["set"].append(int(set))
            for set_prefix in set_prefixes:
                #print(set)
                single_col = f"{set_prefix}_set_{set}"
                
                if single_col in pulse_df.columns:
                    set_data_dict[set_prefix].append(row[single_col])
                else:
                    #print(f"no col found for {single_col}")
                    set_data_dict[set_prefix].append(None)
        
       # print(set_data_dict)

        row_set_df = pd.DataFrame.from_dict(set_data_dict)
        for keep_col in keep_cols:
            row_set_df[keep_col] = row[keep_col]
        
        # decided to do this earlier
        #merged_df = row_set_df.merge(run_info_df_by_pref)

        #print(row_set_df)
        reshaped_df = pd.concat([reshaped_df, row_set_df])

    return reshaped_df


reshaped_e10_pulses_df = reshape_E10_pulse_metrics(merged_pulse_df)

reshaped_e10_pulses_df["soc_post_charge_rounded"] = reshaped_e10_pulses_df["SOC_post_charge_to_target"].apply(lambda f: round(f, 2))
reshaped_e10_pulses_df["soc_post_discharge_rounded"] = reshaped_e10_pulses_df["SOC_post_discharge_to_target"].apply(lambda f: round(f, 2))
reshaped_e10_pulses_df["soc_post_charge_rounded_str"] = reshaped_e10_pulses_df["soc_post_charge_rounded"].apply(lambda f: str(f))
reshaped_e10_pulses_df["soc_post_discharge_rounded_str"] = reshaped_e10_pulses_df["soc_post_discharge_rounded"].apply(lambda f: str(f))


# %%

# ============================================================================= 
# ====================== Plot RPT Pulse ASR Metrics ==========================
# =============================================================================

reshaped_e10_pulses_df['experiment']=reshaped_e10_pulses_df['sample_name'].str[0:6]
reshaped_e10_pulses_df['process']=reshaped_e10_pulses_df['sample_name'].str[0:8]


# make boxplot of Vmin slope by group_by_col
fig = px.box(reshaped_e10_pulses_df[reshaped_e10_pulses_df.soc_post_discharge_rounded==0.5], x='process', y="RPT_discharge_asr_18sec", title='RPT 50% SOC Discharge ASR', points='all', hover_data=['sample_name'])
fig.update_yaxes(range=[16,26], title='Discharge ASR [Ω cm²]', tickfont=dict(size=24),
                # dtick=0.5,
                    titlefont=dict(size=26), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')
fig.update_xaxes(title='', tickfont=dict(size=18),
                # dtick=0.5,
                                #   categoryorder='array', categoryarray=['1wt% TTFEP',
                                #                                                 '2wt% TTFEP',
                                #                                                 # '3wt% TTFEP'
                                #                                                 ],
                    titlefont=dict(size=26), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')

colors = px.colors.qualitative.Plotly

fig.update_traces(line=dict(color="black"), fillcolor=colors[0])

# add line at 23 ohm cm2 and annotate with "23 Ω cm²"

fig.add_shape(
    type="line",
    x0=-0.5,
    x1=9.8,
    y0=23,
    y1=23,
    # make dashed grey line
    line=dict(color="grey", width=2, dash="dash"),
)

fig.add_annotation(
    x=9,
    y=23.5,
    text="23 Ω cm²",
    showarrow=False,
    font=dict(size=20),
)

fig.update_layout(
    # change plot background color to white
    plot_bgcolor='white',
    # add y axis grid lines
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    autosize=False,
    width=900,
    height=700,
    font = dict(size = 20)
)


fig.show(renderer="browser")


# %%

# ============================================================================= 
# ==================== Calculate GED based on Pouch mass=======================
# =============================================================================


# mass_tck = pd.read_csv("QSC020_mass_thickness.csv")

# data = df_master[df_master.cell_tier_group.str.contains('Tier 1a|Tier 1b|Tier 2')]
data=df_master.copy()
data = data[(data["C/3 Count"] == 1) | (data['samplename'].str.contains('APD229')) ]


data = data.merge(mass_tck, on="samplename", how="left")

data = data[~data.samplename.str.contains('QSC020A[C,E,G]')]

data["DischargeEnergy_Co3_first"] = data["DischargeEnergy_Co3_first"] / 1000

data["GED"] = data["DischargeEnergy_Co3_first"] / data["Mass (kg)"]

data=data.sort_values('process')

fig = px.box(data, x='process', y="GED", title='C/3 Discharge Energy', points='all', hover_data=['samplename'])
fig.update_yaxes(title='Discharge Energy (Wh/kg)', tickfont=dict(size=24),
                # dtick=0.5,
                    titlefont=dict(size=26), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')
fig.update_xaxes(title='', tickfont=dict(size=22),
                # dtick=0.5,
                    titlefont=dict(size=26), mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey')

fig.update_traces(line=dict(color="black"), fillcolor=colors[0])

fig.update_layout(
    # change plot background color to white
    plot_bgcolor='white',
    # add y axis grid lines
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    autosize=False,
    width=900,
    height=700,
    font = dict(size = 20)
)


fig.show(renderer="browser")
# %%
data.to_clipboard(index=False)
# %%

# ============================================================================= 
# ====================== Plot 22L Screen Metrics ==============================
# =============================================================================


data=df_master.copy()

data["Co3_Ratio"]=data["AMSDischargeCapactiy_Co3_first"]/data["AMSDischargeCapactiy_Co3_last"]

data.loc[data["samplename"].str.contains('APD229BC-PS00-02'), "1C Count"] = 1

data = data[data["1C Count"] == 1]
data = data[data["cell_tier_group"].str.contains('Tier 1a|Tier 1b')]



import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Columns you want to plot
columns_to_plot = ["AMSDischargeCapactiy_Co3_last", "MedDischargeASR_1C", "dVdt_delta_1C", 
                   "AMSDischargeCapactiy_1C", "dVdt_1C_min", "CE_1C_min"]

# Colors for the boxplots, you can adjust this as needed
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create subplots
fig = make_subplots(rows=2, cols=3, subplot_titles=columns_to_plot)

# Loop through the columns and add a boxplot to each subplot
for i, column in enumerate(columns_to_plot):
    row = i // 3 + 1
    col = i % 3 + 1
    fig.add_trace(
        go.Box(x=data['Condition'],y=data[column], name=column, marker_color=colors[i % len(colors)], 
               boxpoints='all',  # Show all points
               jitter=0.3,  # Spread out points for better visibility
               pointpos=-1.5,  # Position points symmetrically around the center
               hoverinfo='y+text',  # Customize hover information
               text=data['samplename'],  # This will set hover text to the 'samplename' column  # Show only the y-axis value when hovering
               showlegend=False),
        row=row, col=col
    )
    # Update the y-axis title for each subplot
    fig.update_yaxes(title_text=column, tickfont=dict(size=16), titlefont=dict(size=16),
                     mirror=True, ticks='outside', showline=True, linewidth=2, linecolor='grey', title_standoff=5,
                     row=row, col=col)
    fig.update_xaxes(categoryorder='array', categoryarray=['Validation', 'Shippment<br>Candidates<br>Tier 1a', 'Shippment<br>Candidates<br>Tier 1b'])

# Update x axes titles
fig.update_xaxes(title_text='', tickfont=dict(size=16),
                 titlefont=dict(size=16), mirror=True, ticks='outside', showline=True, 
                 linewidth=2, linecolor='grey')

# Adjust layout
fig.update_layout(
    plot_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    autosize=False,
    width=1450,
    height=800,
    font=dict(size=20),
    title='Boxplots for Multiple Metrics'
)

# Show figure
fig.show()
# %%
# %%
# ====================================================================================
# ======================        Genealogy        =====================================
# ====================================================================================


df_genealogy22L = genealogy.get_genealogy_6L(search, conn)
df_genealogy22L.rename(columns={"2L_cell_id": "US_id"}, inplace=True)
df_genealogy22L.dropna(subset=["US_id"], inplace=True)

# get genealogy data for the unit cell IDs
df_genealogy2L = genealogy.get_genealogy_unitcell(df_genealogy22L["US_id"])
df_genealogy22L = df_genealogy22L.merge(df_genealogy2L, on="US_id", how="left")

# %%

df_master_summary = df_master[(df_master['C/3 Count']==1) & df_master['samplename'].str.contains('APD253') | df_master['samplename'].str.contains('APD253AB-PS02-01')]

df_master_summary = df_master_summary[['samplename', 'AMSDischargeCapactiy_1C', 'AMSDischargeCapactiy_Co3_first', 'AMSDischargeCapactiy_Co3_last',
                                 'DischargeEnergy_Co3_first', 'DischargeEnergy_Co3_last', 'AMSChargeCapacity_Co3_RPT', 'MedDischargeASR_1C', 'dVdt_delta_1C',
                                   'cell_build_date', 'cell_build_WW', 'cell_tier_group']]

df_master_summary.to_clipboard(index=False)

# df_genealogy22L.to_clipboard(index=False)
# %%
######## QUERY 2L ELECTRICAL METRICS AND JOIN ########

# get yield and electrical metrics for the unit cell IDs
df_electrical_yield_metrics = uceym_rel.get_electrical_yield_and_metrics(
    df_genealogy22L["US_id"]
)

df_electrical_yield_metrics.rename(columns={"cell_build_date": "cell_build_date_2L",
                                            "cell_build_WW": "cell_build_WW_2L",
                                            "MedDischargeASR_1C": "MedDischargeASR_1C_2L",
                                            "AMSDischargeCapactiy_1C": "AMSDischargeCapactiy_1C_2L",
                                            "dVdt_1C": "dVdt_1C_2L",
                                                    }, inplace=True)
# df_testdata = df_genealogy22L.merge(df_electrical_yield_metrics, on="US_id")

df_summary_metrics = df_electrical_yield_metrics[['US_id', 'cell_build_date_2L', 'cell_build_WW_2L', 'AMSDischargeCapactiy_1C_2L', 'MedDischargeASR_1C_2L', 'dVdt_1C_2L']]

df_genealogy22L = df_genealogy22L.merge(df_summary_metrics, on="US_id", how="left")


# %%
df_master2 = pd.merge(df_master, df_genealogy22L, left_on="samplename", right_on="6L_cell_id", how="left")

# %%

data = df_master2[df_master2.cell_tier_group.str.contains('Tier 1a|Tier 1b')]


data.loc[data["samplename"].str.contains('APD229BC-PS00-02'), "1C Count"] = 1

data = data[data["1C Count"] == 1]
data = data[data["cell_tier_group"].str.contains('Tier 1a|Tier 1b')]
data = data[~data.samplename.str.contains('APD229BH|QSC020AF-PS00-01')]

data=data.sort_values('process')

# Assuming 'data' is your DataFrame
# Step 1: Create a unique DataFrame for the markers
unique_asr = data.drop_duplicates(subset=['samplename'])[['samplename', 'MedDischargeASR_1C']]

# Step 2: Create the boxplot
fig = go.Figure()

fig.add_trace(
    go.Box(
        y=data['MedDischargeASR_1C_2L'],
        x=data['samplename'],
        name='2L ASR',
        boxpoints='all',  # Show all points
        jitter=0.5,  # Spread out points for better visibility
        pointpos=0,  # Center points under the box
        marker_color='blue'  # Color for individual points
    )
)

# Step 3: Overlay the markers
fig.add_trace(
    go.Scatter(
        x=unique_asr['samplename'],
        y=unique_asr['MedDischargeASR_1C'],
        mode='markers',
        marker=dict(color='red', size=10),  # Large red markers
        name='Overall ASR'
    )
)

# Update layout
fig.update_layout(
    title='Comparison of 2L and Overall 22L Resistance per Sample',
    xaxis_title='Sample Name',
    yaxis_title='Resistance (Ω.cm²)',
    showlegend=True,
    plot_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    autosize=False,
    width=1350,
    height=600,
    font=dict(size=16)
)

# Show figure
fig.show()


# %%
