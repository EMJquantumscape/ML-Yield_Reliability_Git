# %%
import pandas as pd
from qsdc.client import Client


# %%
def combine_SAs(df_gen):
    df_SA00 = df_gen[
        df_gen["SA_id"].notna() & df_gen["SA_id"].str.contains("SA0[0,2,4,6,8]")
    ]
    df_SA00 = df_SA00.rename(
        columns={
            "SA_id": "SA00_id",
            "DP_id": "SA00_DP_id",
            "film_HT_id": "SA00_film_HT_id",
            "film_LC_id": "SA00_film_LC_id",
            "film_FS_id": "SA00_film_FS_id",
            "film_HT_time": "SA00_film_HT_time",
            "film_LC_time": "SA00_film_LC_time",
            "film_FS_time": "SA00_film_FS_time",
        }
    )

    df_SA01 = df_gen[
        df_gen["SA_id"].notna() & df_gen["SA_id"].str.contains("SA0[1,3,5,7,9]")
    ]
    df_SA01 = df_SA01.rename(
        columns={
            "SA_id": "SA01_id",
            "DP_id": "SA01_DP_id",
            "film_HT_id": "SA01_film_HT_id",
            "film_LC_id": "SA01_film_LC_id",
            "film_FS_id": "SA01_film_FS_id",
            "film_HT_time": "SA01_film_HT_time",
            "film_LC_time": "SA01_film_LC_time",
            "film_FS_time": "SA01_film_FS_time",
        }
    )

    return df_SA00.merge(
        df_SA01[
            [
                "2L_cell_id",
                "SA01_id",
                "SA01_DP_id",
                "SA01_film_HT_id",
                "SA01_film_LC_id",
                "SA01_film_FS_id",
                "SA01_film_HT_time",
                "SA01_film_LC_time",
                "SA01_film_FS_time",
            ]
        ],
        on=["2L_cell_id"],
        how="outer",
    )


def get_genealogy_2L(search, conn):
    df_gen = pd.read_sql_query(
        f"""
    SELECT * FROM (
        SELECT 
        sample.displayname AS SA_id,
        film_1.displayname AS film_HT_id,
        film_2.displayname AS film_LC_id,
        film_3.displayname AS film_FS_id,
        # film_4.displayname AS film_FS_id_WT,
        film1_p.ended AS film_HT_time,
        film2_p.ended AS film_LC_time,
        film3_p.ended AS film_FS_time

        FROM sample_link
        INNER JOIN device_structure film_1 ON sample_link.idsample_from = film_1.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process film1_p ON film1_p.idprocess = film_1.idprocess_createdby
        LEFT JOIN process sa_process ON sa_process.idprocess = sample.idprocess_createdby
        LEFT JOIN component AS film_1_component ON sample_link.idcomponent = film_1_component.idcomponent 
        LEFT JOIN component AS film_2_component ON film_1.idcomponent_assigned = film_2_component.idcomponent
        LEFT JOIN device_structure as film_2 on film_2_component.iddevice = film_2.iddevice
        LEFT JOIN process film2_p ON film2_p.idprocess = film_2.idprocess_createdby
        LEFT JOIN component AS film_3_component ON film_2.idcomponent_assigned = film_3_component.idcomponent
        LEFT JOIN device_structure as film_3 on film_3_component.iddevice = film_3.iddevice
        LEFT JOIN process film3_p ON film3_p.idprocess = film_3.idprocess_createdby
        # LEFT JOIN component AS film_4_component ON film_3.idcomponent_assigned = film_3_component.idcomponent
        # LEFT JOIN device_structure as film_4 on film_4_component.iddevice = film_4.iddevice
        # LEFT JOIN process film4_p ON film4_p.idprocess = film_4.idprocess_createdby

        WHERE sample.displayname REGEXP '{search}' 
        # AND film_4.displayname REGEXP '(.+)-' 
        AND sample.displayname REGEXP '(.+)-SA'
    ) sa_to_film

    LEFT JOIN (
        SELECT 
        sample.displayname AS DP_id,
        component.displayname AS SA_id

        FROM sample_link
        INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process dp_process ON dp_process.idprocess = sample.idprocess_createdby

        WHERE sample.displayname REGEXP '(.+)-DP'
    ) dp_to_sa USING(SA_id)



    LEFT JOIN (
        SELECT 
        sample.displayname AS CS_id,
        component.displayname AS DP_id

        FROM sample_link
        INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby

        WHERE sample.displayname REGEXP '(.+)-CS'
    ) cs_to_dp USING(DP_id)

    LEFT JOIN (
        SELECT 
        sample.displayname AS CS_id,
        component.displayname AS SK_id
    

        FROM sample_link
        INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby
        WHERE component.displayname REGEXP '(.+)-SK'

    ) CS_to_SK USING(CS_id)

    LEFT JOIN (
        SELECT 
        sample.displayname AS SK_id,
        component.displayname AS CAT_id,
        cat_process.description AS CAT_process,
        cat_tool.displayname AS CAT_tool

        FROM sample_link
        INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process cat_process ON cat_process.idprocess = sample.idprocess_createdby
        LEFT JOIN tool cat_tool ON cat_tool.idtool = cat_process.idtool_requested
        WHERE component.displayname REGEXP 'CAT|MCT'

    ) SK_to_CAT USING(SK_id)

    LEFT JOIN (
        SELECT
        sample.displayname AS 2L_cell_id,
        sample.results AS 2L_Cell_Results,
        component.displayname AS CS_id,
        cell_material.displayname AS cell_material,
        fx_tool.displayname AS fx_tool,
        fx_tool_recipe.name AS fx_recipe,
        cell_process.ended AS cell_build_time

        FROM sample_link
        INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
        LEFT JOIN tool fx_tool ON fx_tool.idtool = cell_process.idtool_requested
        LEFT JOIN tool_recipe fx_tool_recipe ON fx_tool_recipe.idtool_recipe = cell_process.idtool_recipe

        INNER JOIN component cell_component ON cell_component.idprocess = cell_process.idprocess
        LEFT JOIN material cell_material ON cell_material.idmaterial = cell_component.idmaterial

        WHERE 
        sample.displayname REGEXP '(.+)-(US|PS)' 
        AND (cell_material.displayname LIKE '%%Foam%%' OR cell_material.displayname LIKE '%%Rubber%%')
    ) ust_to_cs USING(CS_id)

    LEFT JOIN (
        SELECT
        CS2.displayname AS 6L_cell_id,
        US.displayname AS 2L_cell_id

        FROM device_structure CS2
        LEFT JOIN sample_link ON CS2.iddevice=sample_link.idsample_to
        LEFT JOIN device_structure US ON US.iddevice=sample_link.idsample_from
        LEFT JOIN sample_status ss ON ss.idsample_status = CS2.idsample_status
        LEFT JOIN process CS2_process ON CS2_process.idprocess = CS2.idprocess_createdby

    ) US_to_CH USING(2L_cell_id)

    """,
        conn,
    )

    df_gen = df_gen.drop_duplicates(subset="SA_id")

    df_gen = df_gen[df_gen["2L_cell_id"].notnull()]

    df_gen = combine_SAs(df_gen)

    return df_gen


# %%

# from qsdc.client import Client
# import pandas as pd

# conn = Client().get_mysql_engine()
# gen = get_genealogy_2L('APD224A[U-Z]|APD224B', conn)


# %%
def get_genealogy_6L(search, conn):
    df_gen = pd.read_sql_query(
        f"""

        SELECT * FROM (
            SELECT
            US_6L.displayname AS 6L_cell_id,
            US_2L.displayname AS 2L_cell_id

            FROM device_structure US_6L
            LEFT JOIN sample_link ON US_6L.iddevice=sample_link.idsample_to
            LEFT JOIN device_structure US_2L ON US_2L.iddevice=sample_link.idsample_from
            LEFT JOIN sample_status ss ON ss.idsample_status = US_6L.idsample_status
            LEFT JOIN process US_6L_process ON US_6L_process.idprocess = US_6L.idprocess_createdby

            WHERE US_6L.displayname REGEXP '{search}' 
            AND US_6L.displayname REGEXP '(.+)-(US|PS)'
            AND (US_6L_process.layer_count = 3 
            OR US_6L_process.layer_count = 1)
        ) US_6L_to_US_2L

        LEFT JOIN (
            SELECT
            sample.displayname AS 2L_cell_id,
            component.displayname AS CS_id     

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-CS'

        ) US_to_CS USING(2L_cell_id)

        LEFT JOIN (
            SELECT 
            sample.displayname AS CS_id,
            component.displayname AS SA_id

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-SA'

        ) CS_to_SA USING(CS_id)

        LEFT JOIN (
            SELECT 
            sample.displayname AS SA_id,
            film_1.displayname AS film_HT_id,
            film_2.displayname AS film_LC_id,
            film_3.displayname AS film_FS_id

            FROM sample_link
            INNER JOIN device_structure film_1 ON sample_link.idsample_from = film_1.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process film1_p ON film1_p.idprocess = film_1.idprocess_createdby
            LEFT JOIN process sa_process ON sa_process.idprocess = sample.idprocess_createdby
            LEFT JOIN component AS film_1_component ON sample_link.idcomponent = film_1_component.idcomponent 
            LEFT JOIN component AS film_2_component ON film_1.idcomponent_assigned = film_2_component.idcomponent
            LEFT JOIN device_structure as film_2 on film_2_component.iddevice = film_2.iddevice
            LEFT JOIN process film2_p ON film2_p.idprocess = film_2.idprocess_createdby
            LEFT JOIN component AS film_3_component ON film_2.idcomponent_assigned = film_3_component.idcomponent
            LEFT JOIN device_structure as film_3 on film_3_component.iddevice = film_3.iddevice
            LEFT JOIN process film3_p ON film3_p.idprocess = film_3.idprocess_createdby

        ) SA_to_film USING(SA_id)

        """,
        conn,
    )

    df_gen = combine_SAs(df_gen)

    return df_gen[
        [
            "6L_cell_id",
            "2L_cell_id",
            "CS_id",
            "SA00_id",
            "SA00_film_HT_id",
            "SA00_film_LC_id",
            "SA00_film_FS_id",
            "SA01_id",
            "SA01_film_HT_id",
            "SA01_film_LC_id",
            "SA01_film_FS_id",
        ]
    ]


def get_genealogy_6L_v2(search, conn):
    df_gen = pd.read_sql_query(
        f"""

        SELECT * FROM (
            SELECT
            US_6L.displayname AS 6L_US_id,
            CS_6L.displayname AS 6L_CS_id

            FROM device_structure US_6L
            LEFT JOIN sample_link ON US_6L.iddevice=sample_link.idsample_to
            LEFT JOIN device_structure CS_6L ON CS_6L.iddevice=sample_link.idsample_from
            LEFT JOIN sample_status ss ON ss.idsample_status = US_6L.idsample_status
            LEFT JOIN process US_6L_process ON US_6L_process.idprocess = US_6L.idprocess_createdby

            WHERE US_6L.displayname REGEXP '{search}' 
            # AND US_6L.displayname REGEXP '(.+)-[U,P]S'
            # AND (US_6L_process.layer_count = 3 
            # OR US_6L_process.layer_count = 1)
        ) US_6L_to_CS_6L

        # LEFT JOIN (
        #     SELECT
        #     sample.displayname AS 6L_CS_id,
        #     component.displayname AS 2L_US_id     

        #     FROM sample_link
        #     INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
        #     INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
        #     LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
        #     LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
        #     WHERE component.displayname REGEXP '(.+)-US'

        # ) CS_6L_to_US_2L USING(6L_CS_id)

        """,
        conn,
    )

    return df_gen


def get_genealogy_24L(search, conn):
    df_gen = pd.read_sql_query(
        f"""
        SELECT * FROM (
            SELECT
            US_24L.displayname AS 24L_cell_id,
            CS_24L.displayname AS 24L_CS_id

            FROM device_structure US_24L
            LEFT JOIN sample_link ON US_24L.iddevice=sample_link.idsample_to
            LEFT JOIN device_structure CS_24L ON CS_24L.iddevice=sample_link.idsample_from
            LEFT JOIN sample_status ss ON ss.idsample_status = US_24L.idsample_status
            LEFT JOIN process US_24L_process ON US_24L_process.idprocess = US_24L.idprocess_createdby

            WHERE US_24L.displayname REGEXP '{search}' 
            AND US_24L.displayname REGEXP '(.+)-PS'
            AND US_24L.idsample_status = 2
        ) US_24L_to_CS_24L 

        LEFT JOIN (
            SELECT
            sample.displayname AS 24L_CS_id,
            component.displayname AS 6L_cell_id     

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-US'

        ) CS_24L_to_US_6L USING(24L_CS_id)
  
        LEFT JOIN (
            SELECT
            sample.displayname AS 6L_cell_id,
            component.displayname AS 2L_cell_id     

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-US'
        ) US_6L_to_US_2L USING(6L_cell_id)

        LEFT JOIN (
            SELECT
            sample.displayname AS 2L_cell_id,
            component.displayname AS CS_id     

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cell_process ON cell_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-CS'

        ) US_to_CS USING(2L_cell_id)
        
        LEFT JOIN (
            SELECT 
            sample.displayname AS CS_id,
            component.displayname AS SK_id

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-SK'

        ) CS_to_SK USING(CS_id)
        
        LEFT JOIN (
            SELECT 
            sample.displayname AS SK_id,
            component.displayname AS CAT_id

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP 'CAT'

        ) SK_to_CAT USING(SK_id)

        LEFT JOIN (
            SELECT 
            sample.displayname AS CS_id,
            component.displayname AS SA_id

            FROM sample_link
            INNER JOIN device_structure component ON sample_link.idsample_from = component.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process cs_process ON cs_process.idprocess = sample.idprocess_createdby
            WHERE component.displayname REGEXP '(.+)-SA'

        ) CS_to_SA USING(CS_id)

        LEFT JOIN (
            SELECT 
            sample.displayname AS SA_id,
            film_1.displayname AS film_HT_id,
            film_2.displayname AS film_LC_id,
            film_3.displayname AS film_FS_id

            FROM sample_link
            INNER JOIN device_structure film_1 ON sample_link.idsample_from = film_1.iddevice
            INNER JOIN device_structure sample ON sample_link.idsample_to = sample.iddevice
            LEFT JOIN sample_status ss ON ss.idsample_status = sample.idsample_status
            LEFT JOIN process film1_p ON film1_p.idprocess = film_1.idprocess_createdby
            LEFT JOIN process sa_process ON sa_process.idprocess = sample.idprocess_createdby
            LEFT JOIN component AS film_1_component ON sample_link.idcomponent = film_1_component.idcomponent 
            LEFT JOIN component AS film_2_component ON film_1.idcomponent_assigned = film_2_component.idcomponent
            LEFT JOIN device_structure as film_2 on film_2_component.iddevice = film_2.iddevice
            LEFT JOIN process film2_p ON film2_p.idprocess = film_2.idprocess_createdby
            LEFT JOIN component AS film_3_component ON film_2.idcomponent_assigned = film_3_component.idcomponent
            LEFT JOIN device_structure as film_3 on film_3_component.iddevice = film_3.iddevice
            LEFT JOIN process film3_p ON film3_p.idprocess = film_3.idprocess_createdby

        ) SA_to_film USING(SA_id)

        """,
        conn,
    )

    try:
        df_gen = df_gen[df_gen["6L_cell_id"].notna()]
        df_gen = combine_SAs(df_gen)
        return df_gen
    except:
        return df_gen


def pivot_table_with_process_segment_type(df):
    # Initialize a dictionary to hold the counts of each process segment type for a given root sample name
    process_segment_type_counts = {}

    # Initialize an empty dataframe for the pivoted data
    pivoted_df = pd.DataFrame()

    # Set to keep track of the out_sample_names that have already been used for each root_sample_name
    used_out_sample_names = set()

    # Loop through each row in the dataframe to build the pivoted table
    for index, row in df.iterrows():
        root_sample_name = row["root_sample_name"]
        process_segment_type = row["process_segment_type"]
        out_sample_name = row["out_sample_name"]

        # Skip rows where out_sample_name or process_segment_type is NaN
        if pd.isna(out_sample_name) or pd.isna(process_segment_type):
            continue

        # Create a unique key for each root_sample_name and out_sample_name combination
        unique_key = (root_sample_name, out_sample_name)

        # If this unique key has already been added, skip it to avoid repeats
        if unique_key in used_out_sample_names:
            continue

        # Mark this out_sample_name as used for this root_sample_name
        used_out_sample_names.add(unique_key)

        # Create a unique column name for each process_segment_type by appending a count number if needed
        process_segment_type_count_key = (root_sample_name, process_segment_type)
        if process_segment_type_count_key not in process_segment_type_counts:
            process_segment_type_counts[process_segment_type_count_key] = 1
        else:
            process_segment_type_counts[process_segment_type_count_key] += 1
        column_name = f"{process_segment_type} {process_segment_type_counts[process_segment_type_count_key]}"

        # Add the new column if it doesn't exist
        if column_name not in pivoted_df.columns:
            pivoted_df[column_name] = None

        # Add 'root_sample_name' column if it doesn't exist
        if "root_sample_name" not in pivoted_df.columns:
            pivoted_df["root_sample_name"] = None

        # If 'root_sample_name' does not exist in the dataframe, add a new row
        if not pivoted_df["root_sample_name"].isin([root_sample_name]).any():
            new_row = {col: None for col in pivoted_df.columns}
            new_row["root_sample_name"] = root_sample_name
            pivoted_df = pivoted_df.append(new_row, ignore_index=True)

        # Update the corresponding cell in the pivoted dataframe
        pivoted_df.loc[
            pivoted_df["root_sample_name"] == root_sample_name, column_name
        ] = out_sample_name

    # Reorder columns to have 'root_sample_name' as the first column
    columns = ["root_sample_name"] + [
        col for col in pivoted_df.columns if col != "root_sample_name"
    ]
    pivoted_df = pivoted_df[columns]

    return pivoted_df


def get_genealogy_unitcell(sample_ids):
    qs_client = Client()

    # sample_ids = ["APD223AD-US00-01"]

    table_genealogy = qs_client.genealogy.get_all_ancestors(sample_ids)

    pivoted_genealogy_df = pivot_table_with_process_segment_type(table_genealogy)

    rename_dict = {
        "Unit Stack 1": "US_id",
        "Cell Stack 1": "CS_id",
        "Dispense Polymer 1": "DP00_id",
        "Dispense Polymer 2": "DP01_id",
        "Seal Activation 1": "SA00_id",
        "Seal Activation 2": "SA01_id",
        "Wet Treatment 1": "WT00_id",
        "Wet Treatment 2": "WT01_id",
        "Heat Treatment 1": "HT00_id",
        "Heat Treatment 2": "HT01_id",
        "LaserCut 1": "LC00_id",
        "LaserCut 2": "LC01_id",
        "Film Sinter 1": "FS00_id",
        "Film Sinter 2": "FS01_id",
        "Soak 1": "SK_id",
        "Punch 1": "CAT_id",
    }

    pivoted_genealogy_df.rename(columns=rename_dict, inplace=True)

    new_column_names = list(rename_dict.values())
    old_column_names = list(rename_dict.keys())
    missing_old_column_names = [
        col for col in old_column_names if col not in pivoted_genealogy_df.columns
    ]
    missing_new_column_names = [
        col for col in new_column_names if col not in pivoted_genealogy_df.columns
    ]

    for col in missing_new_column_names:
        pivoted_genealogy_df[col] = None

    genealogy_df = pivoted_genealogy_df[new_column_names]

    return genealogy_df


# %%

from qsdc.client import Client
import pandas as pd

conn = Client().get_mysql_engine()
#gen = get_genealogy_2L('APD|QSC', conn)

#sample_ids = ["APD224AZ-US02-04"]
sample_ids = ["QSC020AS-PS00-03"]
gen = get_genealogy_unitcell(sample_ids)


# %%

# from qsdc.client import Client
# qs_client = Client()
# #default group and category parameters are "mfg" and "mfg", respectively


# %%
