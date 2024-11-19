# %%
import pandas as pd
import numpy as np

from qsdc.client import Client
import met_client as app


# %%
# Yield criteria

formation_recipes = [13707, 14212]
formation_charge_capacity_fraction = 1.2
formation_ceiling_rest_voltage = 3.7
formation_dvdt_10minrest = -20
formation_ceiling_hold_time = 7200

screen_strict1C_recipes = [13198, 13709, 13341, 14448, 14924]
strict1C_charge_capacity_fraction = 1.0
strict1C_dvdt = -15
strict1C_ceiling_hold_time = 5000

fast_charge_yield_correction = True
screen_fastcharge_recipes = [13342, 13771, 13775, 14346, 14450, 14608, 14925, 15151]
fastcharge_charge_capacity_fraction = 0.97
fastcharge_dvdt = -15
fastcharge_delta_dvdt = 2
fastcharge_CE = 0.98
fastcharge_ceiling_hold_time = 4500

screen_4C_recipes = [15152]
fourC_charge_capacity_fraction = 0.96
fourC_fastcharge_dvdt = -12
fourC_fastcharge_delta_dvdt = 1
fourC_fastcharge_CE = 0.98
fourC_fastcharge_ceiling_hold_time = 3600

screen_final_1C = [14347]
final1C_charge_capacity_fraction = 1.05
final1C_dvdt = -13
final1C_ceiling_hold_time = 3200

screen_Co3 = [13197, 13708, 13345, 14451]
Co3_charge_capacity_fraction = 1.0
Co3_dvdt = -11

partial_charge_recipes = [13711]

reliability_screen_1C_recipes = [14793, 14803, 14440, 15037]
reliability_screen_1C_charge_capacity_fraction = 0.983
reliability_screen_1C_dvdt = -11

reliability_screen_fast_charge_recipes = [14943, 14445]
reliability_screen_fast_charge_charge_capacity_fraction = 0.95
reliability_screen_fast_charge_dvdt = -11


def query_cycle_data(US_ids):
    # create the quantumscape data client
    qs_client = Client()

    # Query data
    conn = qs_client.get_mysql_engine()

    recipes = (
        formation_recipes
        + screen_Co3
        + screen_strict1C_recipes
        + screen_fastcharge_recipes
        + screen_4C_recipes
        + screen_final_1C
        + reliability_screen_1C_recipes
        + reliability_screen_fast_charge_recipes
        # + partial_charge_recipes
    )

    df_raw = pd.read_sql_query(
        f"""
            SELECT
            device_structure.displayname AS US_id,
            test_run_E12_cycle.VoltagePostCeilingRestEndDVdt * 1E6 AS dvdt,
            test_run_E12_cycle.CapacityChargeActiveMassSpecific AS 'AMSChargeCapacity',
            test_run_E12_cycle.CapacityDischargeActiveMassSpecific AS 'AMSDischargeCapacity',
            test_run_E12_cycle.CapacityDischargeArealDensity AS 'ADDischargeCapacity',
            test_run_E12_cycle.CapacityDischarge AS 'DischargeCapacity',
            test_run_E12_cycle.CapacityCharge AS 'ChargeCapacity',
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
            device_structure.displayname IN ({"'"+"', '".join(US_ids)+"'"})
            AND (device_structure.displayname like '%%-US%%'
            OR device_structure.displayname like '%%-PS%%')
            AND test_run.idtest_recipe IN ({"'"+"', '".join(map(str, recipes))+"'"})
            AND process.layer_count = 1 """,
        conn,
    )

    # group by US_id and check if any TestCycleEnd is null
    df_raw["CycleComplete"] = df_raw["TestCycleEnd"].notnull()
    df_raw["TestComplete"] = df_raw.groupby("US_id")["CycleComplete"].transform(all)

    # remove cycles with AMSChargeCapacity < 1 (OCV cycles)
    df_raw = df_raw[df_raw.AMSChargeCapacity > 1]

    df_raw = df_raw.sort_values(["RunIndex", "CycleIndex"], ascending=True)

    # Determine if the cycle was stopped on short (more efficient method than to query the database)
    df_raw["CumulativeCycle"] = 1
    df_raw.CumulativeCycle = df_raw.groupby("US_id").CumulativeCycle.cumsum()

    df_raw.reset_index(inplace=True)

    df_raw["last_cycle"] = (
        df_raw.groupby("US_id")["CumulativeCycle"].transform(max)
        == df_raw["CumulativeCycle"]
    )

    df_raw["StoppedOnShort"] = (
        df_raw["DischargeCapacity"].isnull()
        & df_raw["last_cycle"]
        & df_raw["TestCycleEnd"].notnull()
    )

    return df_raw


def get_electrical_yield_and_metrics(US_ids):

    df_cyc = query_cycle_data(US_ids)

    df_cyc["batch"] = df_cyc["US_id"].str.slice(stop=13)
    df_cyc["process"] = df_cyc["US_id"].str.slice(stop=8)
    df_cyc["experiment"] = df_cyc["US_id"].str.slice(stop=6)
    df_cyc["project"] = df_cyc["US_id"].str.slice(stop=3)

    df_cyc["4C_UCT_screen"] = df_cyc.groupby("batch")["idtest_recipe"].transform(
        lambda x: x.isin(screen_4C_recipes).any()
    )
    df_cyc["2p5C_UCT_screen"] = df_cyc.groupby("batch")["idtest_recipe"].transform(
        lambda x: x.isin(screen_fastcharge_recipes).any()
    )

    df_cyc = df_cyc.set_index("US_id")

    df_cyc["MedDischargeASR_form"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(formation_recipes),
            ["MedDischargeASR"],
        ]
        .groupby("US_id")
        .first()
    )

    df_cyc["AMSDischargeCapactiy_form"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(formation_recipes),
            ["AMSDischargeCapacity"],
        ]
        .groupby("US_id")
        .first()
    )

    df_cyc["AMSDischargeCapacity_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["AMSDischargeCapacity"]]
        .groupby("US_id")
        .last()
    )

    df_cyc["ADDischargeCapacity_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["ADDischargeCapacity"]]
        .groupby("US_id")
        .last()
    )
    df_cyc["AMSDischargeCapactiy_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes + screen_final_1C),
            ["AMSDischargeCapacity"],
        ]
        .groupby("US_id")
        .last()
    )

    df_cyc["AMSDischargeCapactiy_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["AMSDischargeCapacity"]]
        .groupby("US_id")
        .last()
    )

    df_cyc["ADDischargeCapactiy_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["ADDischargeCapacity"]]
        .groupby("US_id")
        .last()
    )

    df_cyc["DischargeCapactiy_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes + screen_final_1C),
            ["DischargeCapacity"],
        ]
        .groupby("US_id")
        .last()
    )

    df_cyc["DischargeCapacity_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["DischargeCapacity"]]
        .groupby("US_id")
        .last()
    )

    df_cyc["MedDischargeASR_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes + screen_final_1C),
            ["MedDischargeASR"],
        ]
        .groupby("US_id")
        .last()
    )

    df_cyc["MedChargeASR_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes + screen_final_1C),
            ["MedChargeASR"],
        ]
        .groupby("US_id")
        .last()
    )

    df_cyc["ASR_ratio_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(formation_recipes + screen_strict1C_recipes),
            ["ASR_ratio"],
        ]
        .groupby("US_id")
        .last()
    )

    df_cyc["dVdt_delta_fastcharge"] = np.abs(
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_fastcharge_recipes), ["dvdt"]]
        .groupby("US_id")
        .min()
        - df_cyc.loc[df_cyc.idtest_recipe.isin(screen_fastcharge_recipes), ["dvdt"]]
        .groupby("US_id")
        .max()
    )

    df_cyc["dVdt_delta_4C"] = np.abs(
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_4C_recipes), ["dvdt"]]
        .groupby("US_id")
        .min()
        - df_cyc.loc[df_cyc.idtest_recipe.isin(screen_4C_recipes), ["dvdt"]]
        .groupby("US_id")
        .max()
    )

    df_cyc["ChargeCapacity_form"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(formation_recipes),
            ["ChargeCapacity"],
        ]
        .groupby("US_id")
        .max()
    )

    df_cyc["ChargeCapacity_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes), ["ChargeCapacity"]
        ]
        .groupby("US_id")
        .max()
    )

    df_cyc["ChargeCapacity_Co3"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_Co3), ["ChargeCapacity"]]
        .groupby("US_id")
        .max()
    )

    df_cyc["ChargeCapacity_fastcharge"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_fastcharge_recipes), ["ChargeCapacity"]
        ]
        .groupby("US_id")
        .max()
    )

    df_cyc["ChargeCapacity_final1C"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(screen_final_1C), ["ChargeCapacity"]]
        .groupby("US_id")
        .max()
    )

    df_cyc["dVdt_1C"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(screen_strict1C_recipes + screen_final_1C),
            ["dvdt"],
        ]
        .groupby("US_id")
        .min()
    )


    # calculate test start time for each cell
    df_cyc["test_start_time"] = (
        df_cyc.loc[
            df_cyc.idtest_recipe.isin(formation_recipes + formation_recipes),
            ["TestCycleStart"],
        ]
        .groupby("US_id")
        .min()
    )

    # calculate test end time for each cell
    df_cyc["test_end_time"] = (
        df_cyc.loc[df_cyc.idtest_recipe.isin(partial_charge_recipes), ["TestCycleEnd"]]
        .groupby("US_id")
        .min()
    )

    df_cyc["UCT_duration"] = (
        df_cyc["test_end_time"] - df_cyc["test_start_time"]
    ).dt.total_seconds()

    df_cyc = df_cyc.reset_index()




    # If it stopped on short, it failed
    df_cyc["Failed"] = df_cyc["StoppedOnShort"] == 1
    df_cyc["Failed_reliability"] = df_cyc["StoppedOnShort"] == 1

    # Formation
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(formation_recipes))
        & (
            (
                (df_cyc.ChargeCapacityFraction > formation_charge_capacity_fraction)
                | (df_cyc.ChargeCapacityFraction < 0)
            )
            | (df_cyc.CeilingRestVoltage < formation_ceiling_rest_voltage)
            | (
                df_cyc.idtest_recipe.isin(formation_recipes)
                & (df_cyc.dvdt <= formation_dvdt_10minrest)
            )
            | (df_cyc.CeilingHoldTime > formation_ceiling_hold_time)
        ),
        "Failed",
    ] = True

    # C/3 cycle
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(screen_Co3))
        & (
            ((df_cyc.ChargeCapacityFraction > Co3_charge_capacity_fraction))
            | (df_cyc.dvdt <= Co3_dvdt)
        ),
        "Failed",
    ] = True

    # 1C-1C
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(screen_strict1C_recipes))
        & (
            (
                (df_cyc.ChargeCapacityFraction > strict1C_charge_capacity_fraction)
                | (df_cyc.ChargeCapacityFraction < 0.1)
            )
            | (df_cyc.dvdt <= strict1C_dvdt)
            | (df_cyc.CeilingHoldTime > strict1C_ceiling_hold_time)
        ),
        "Failed",
    ] = True

    # Fast charge
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(screen_fastcharge_recipes))
        & (
            (
                (df_cyc.ChargeCapacityFraction > fastcharge_charge_capacity_fraction)
                | (df_cyc.ChargeCapacityFraction < 0.1)
            )
            | (df_cyc.dvdt <= fastcharge_dvdt)
            | ((df_cyc.CE < fastcharge_CE) & (df_cyc.AMSDischargeCapacity > 140))
            | (df_cyc.dVdt_delta_fastcharge > fastcharge_delta_dvdt)
            | (df_cyc.CeilingHoldTime > fastcharge_ceiling_hold_time)
        ),
        "Failed",
    ] = True

    df_cyc[["idtest_recipe", "Failed"]]

    # 4C
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(screen_4C_recipes))
        & (
            (
                (df_cyc.ChargeCapacityFraction > fourC_charge_capacity_fraction)
                | (df_cyc.ChargeCapacityFraction < 0.1)
            )
            | (df_cyc.dvdt <= fourC_fastcharge_dvdt)
            | ((df_cyc.CE < fourC_fastcharge_CE) & (df_cyc.AMSDischargeCapacity > 140))
            | (df_cyc.dVdt_delta_4C > fourC_fastcharge_delta_dvdt)
            | (df_cyc.CeilingHoldTime > fourC_fastcharge_ceiling_hold_time)
        ),
        "Failed",
    ] = True

    # Final 1C
    df_cyc.loc[
        (df_cyc.idtest_recipe.isin(screen_final_1C))
        & (
            (
                (df_cyc.ChargeCapacityFraction > final1C_charge_capacity_fraction)
                | (df_cyc.ChargeCapacityFraction < 0.1)
            )
            | (df_cyc.dvdt <= final1C_dvdt)
            | (df_cyc.CeilingHoldTime > final1C_ceiling_hold_time)
        ),
        "Failed",
    ] = True



    # Reliability Screen 1C
    df_cyc.loc[
    (df_cyc.idtest_recipe.isin(reliability_screen_1C_recipes))
    & (
        (df_cyc.ChargeCapacityFraction > reliability_screen_1C_charge_capacity_fraction)
        | (df_cyc.dvdt <= reliability_screen_1C_dvdt)
    ), 
    "Failed_reliability",
    ] = True

    # Reliability Screen Fast Charge
    df_cyc.loc[
    (df_cyc.idtest_recipe.isin(reliability_screen_fast_charge_recipes))
    & (
        (df_cyc.ChargeCapacityFraction > reliability_screen_fast_charge_charge_capacity_fraction)
        | (df_cyc.dvdt <= reliability_screen_fast_charge_dvdt)
    ),
    "Failed_reliability",
    ] = True


    





    df_cyc = df_cyc.merge(
    df_cyc[["US_id", "Failed", "Failed_reliability"]].groupby("US_id").max(),
    suffixes=["", "_any"],
    right_index=True,
    left_on="US_id",
    )

    if fast_charge_yield_correction:
        df_fastcharge = (
            df_cyc.loc[
                df_cyc.idtest_recipe.isin(screen_fastcharge_recipes),
                ["US_id", "idtest_recipe", "Failed"],
            ]
            .groupby(["US_id", "idtest_recipe"])
            .max()
        )
        df_fastcharge.reset_index(inplace=True)
        df_fastcharge = df_fastcharge[["US_id", "Failed"]].rename(
            columns={"Failed": "Failed_fastcharge"}
        )

        df_4C = (
            df_cyc.loc[
                df_cyc.idtest_recipe.isin(screen_4C_recipes),
                ["US_id", "idtest_recipe", "Failed"],
            ]
            .groupby(["US_id", "idtest_recipe"])
            .max()
        )
        df_4C.reset_index(inplace=True)
        df_4C = df_4C[["US_id", "Failed"]].rename(columns={"Failed": "Failed_4C"})

        df_1C = (
            df_cyc.loc[
                df_cyc.idtest_recipe.isin(screen_final_1C),
                ["US_id", "idtest_recipe", "Failed"],
            ]
            .groupby(["US_id", "idtest_recipe"])
            .max()
        )
        df_1C.reset_index(inplace=True)
        df_1C = df_1C[["US_id", "Failed"]].rename(columns={"Failed": "Failed_1C"})

        df_Co3 = (
            df_cyc.loc[
                df_cyc.idtest_recipe.isin(screen_Co3),
                ["US_id", "idtest_recipe", "Failed"],
            ]
            .groupby(["US_id", "idtest_recipe"])
            .max()
        )
        df_Co3.reset_index(inplace=True)
        df_Co3 = df_Co3[["US_id", "Failed"]].rename(columns={"Failed": "Failed_Co3"})

        df_cyc = df_cyc.merge(df_fastcharge, on="US_id", how="left")
        df_cyc = df_cyc.merge(df_4C, on="US_id", how="left")
        df_cyc = df_cyc.merge(df_1C, on="US_id", how="left")
        df_cyc = df_cyc.merge(df_Co3, on="US_id", how="left")

        # if Failed_fastcharge is not nan, then Failed_any = Failed_fastcharge
        df_cyc.loc[df_cyc.Failed_fastcharge.notnull(), "Failed_any"] = df_cyc.loc[
            df_cyc.Failed_fastcharge.notnull(), "Failed_fastcharge"
        ]

        # if Failed_4C is not nan, then Failed_any = Failed_4C | Failed_fastcharge
        df_cyc.loc[(df_cyc.Failed_4C.notnull()), "Failed_any"] = (
            df_cyc.loc[(df_cyc.Failed_4C.notnull()), "Failed_any"]
            | df_cyc.loc[(df_cyc.Failed_4C.notnull()), "Failed_4C"]
        )

        # if Failed_1C is not nan, then Failed_any = Failed_1C | Failed_fastcharge
        df_cyc.loc[(df_cyc.Failed_1C.notnull()), "Failed_any"] = (
            df_cyc.loc[(df_cyc.Failed_1C.notnull()), "Failed_any"]
            | df_cyc.loc[(df_cyc.Failed_1C.notnull()), "Failed_1C"]
        )

        # if Failed_Co3 is not nan, then Failed_any = Failed_Co3 | Failed_1C | Failed_fastcharge
        df_cyc.loc[(df_cyc.Failed_Co3.notnull()), "Failed_any"] = (
            df_cyc.loc[(df_cyc.Failed_Co3.notnull()), "Failed_any"]
            | df_cyc.loc[(df_cyc.Failed_Co3.notnull()), "Failed_Co3"]
        )

    df_cyc["ShortEvent"] = df_cyc.Failed_any | df_cyc.Failed_reliability

    df_cyc_screen = pd.concat(
    [
        df_cyc.loc[
            (df_cyc.ShortEvent == True)
            & ((df_cyc.Failed == True) | (df_cyc.Failed_reliability == True))
        ]
        .groupby("US_id")
        .first(),
        df_cyc.loc[(df_cyc.ShortEvent == False)].groupby("US_id").last(),
    ]
    )
    df_cyc_screen["EventCycle"] = df_cyc_screen.CumulativeCycle
    df_cyc_screen = df_cyc_screen[~df_cyc_screen.index.duplicated()]


    df_sample = pd.DataFrame(df_cyc["US_id"].unique(), columns=["US_id"]).join(
        df_cyc_screen, on="US_id", how="left"
    )

    df_sample["Build Count"] = 1

    df_sample["Formation Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(formation_recipes))
            )
        )
        & (df_sample.CumulativeCycle < 3),
        0,
        1,
    )

    df_sample["1C Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(screen_strict1C_recipes))
            )
            | (df_sample["Formation Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 8),
        0,
        1,
    )

    df_sample["Fast Charge Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(screen_fastcharge_recipes))
            )
            | (df_sample["1C Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 12),
        0,
        1,
    )

    df_sample["4C Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(screen_4C_recipes))
            )
            | (df_sample["Fast Charge Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 12),
        0,
        1,
    )

    df_sample["Final 1C Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(screen_final_1C))
            )
            | (df_sample["4C Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 14),
        0,
        1,
    )

    df_sample["C/3 Count"] = np.where(
        (
            (
                (df_sample.ShortEvent == True)
                & (df_sample.idtest_recipe.isin(screen_Co3))
            )
            | (df_sample["Final 1C Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 14),
        0,
        1,
    )

    df_sample["Yield Count"] = df_sample["C/3 Count"]


    df_sample["Reliability Test Count"] = 0

    df_sample.loc[df_sample.idtest_recipe.isin(reliability_screen_1C_recipes+reliability_screen_fast_charge_recipes), "Reliability Test Count"] = 1

    df_sample["Reliability Screen 1C Count"] = np.where(
        (
            ((df_sample.ShortEvent == True) & (df_sample.idtest_recipe.isin(reliability_screen_1C_recipes)))
            | (df_sample["C/3 Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 35),
        0,
        1,
    )

    df_sample["Reliability Screen Fast Charge Count"] = np.where(
        (
            ((df_sample.ShortEvent == True) & (df_sample.idtest_recipe.isin(reliability_screen_fast_charge_recipes)))
            | (df_sample["Reliability Screen 1C Count"] == 0)
        )
        & (df_sample.CumulativeCycle < 35),
        0,
        1,
    )



    df_sample.reset_index(inplace=True)

    df_sample["1C Count"] = df_sample["1C Count"] * df_sample["batch"].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(screen_strict1C_recipes)
        .astype(int)
        .max()
    )

    df_sample["Fast Charge Count"] = df_sample["Fast Charge Count"] * df_sample[
        "batch"
    ].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(screen_fastcharge_recipes)
        .astype(int)
        .max()
    )

    df_sample["4C Count"] = df_sample["4C Count"] * df_sample["batch"].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(screen_4C_recipes)
        .astype(int)
        .max()
    )

    df_sample["Final 1C Count"] = df_sample["Final 1C Count"] * df_sample[
        "batch"
    ].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(screen_final_1C)
        .astype(int)
        .max()
    )

    df_sample["C/3 Count"] = df_sample["C/3 Count"] * df_sample["batch"].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(screen_Co3)
        .astype(int)
        .max()
    )


    df_sample["Reliability Screen 1C Count"] = df_sample["Reliability Screen 1C Count"] * df_sample["batch"].apply(
    lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
    .isin(reliability_screen_1C_recipes)
    .astype(int)
    .max()
)

    df_sample["Reliability Screen Fast Charge Count"] = df_sample["Reliability Screen Fast Charge Count"] * df_sample["batch"].apply(
        lambda x: df_cyc[df_cyc["batch"] == x]["idtest_recipe"]
        .isin(reliability_screen_fast_charge_recipes)
        .astype(int)
        .max()
    )

    df_sample["cell_build_date"] = df_sample.groupby("process")[
        "cell_build_time"
    ].transform("min")
    df_sample["cell_build_WW"] = (
        df_sample["cell_build_date"].dt.isocalendar().year.astype(str)
        + "WW"
        + df_sample["cell_build_date"].dt.isocalendar().week.astype(str)
    )

    df_sample["cell_build_date"] = df_sample["cell_build_date"].dt.date

    df_sample = df_sample.drop(
        columns=[
            "level_0",
            "index",
            "dvdt",
            "AMSChargeCapacity",
            "AMSDischargeCapacity",
            "ADDischargeCapacity",
            "DischargeCapacity",
            "ChargeCapacity",
            "ChargeCapacityFraction",
            "CE",
            "MedChargeASR",
            "MedDischargeASR",
            "ASR_ratio",
            "CeilingHoldTime",
            "CeilingRestVoltage",
            "CycleIndex",
            "RunIndex",
            "idtest_recipe",
            "TestCycleStart",
            "TestCycleEnd",
            "HardShort",
            "idtest_run_E12_cycle",
            "ProcessorAssumedCapacity",
            "OCVInitial",
            "ProcessDescription",
            "cell_build_time",
            "CycleComplete",
            "Failed",
            "Failed_any",
            "Failed_fastcharge",
            "Failed_4C",
            "Failed_1C",
            "Failed_Co3",
        ],
        errors="ignore",
    )

    return df_sample


# %%
