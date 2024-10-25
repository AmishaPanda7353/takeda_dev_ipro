import pandas as pd
import numpy as np
from  sqlalchemy import create_engine, text
from utils import S3, clean_column_names, connect_to_mysql_db, load_table_to_db
from core.utils.read_config import initialize_config

def load_offline_reports(config):
    """
    This functions loads offline reports to database
    """
    
    # get the required config parameters
    market = config.load_tables.market.name
    round = config.load_tables.market.round
    round_id = config.load_tables.market.round_id
    round_name = config.load_tables.market.round_name
    round_description = config.load_tables.market.round_description
    s3_bucket = config.load_tables.path.s3_bucket

    # reports_path = config.load_tables.path.reports_path
    
    recommendation_playbook_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.recommendation_playbook_folder}/"
    modelled_items_reason_code = "Item_reason_code"
    non_modelled_items_reason_code = "Linked_Item_reason_code"
    
    conn = connect_to_mysql_db(config)
    s3 = S3(s3_bucket)
    
    query1 = text("SELECT * FROM store_master")
    query2 = text("SELECT * FROM store_auto_premium_formula")
    query3 = text("SELECT * FROM user_profile")
    query4 = text("SELECT * FROM opt_BaseItemGroups")
    query5 = text("SELECT * FROM opt_instore_delivery_item_mapping")
    
    store_master = pd.read_sql_query(query1, con=conn)
    store_auto_premium_formula = pd.read_sql_query(query2, con=conn)
    user_profile = pd.read_sql_query(query3, con=conn)
    opt_BaseItemGroups = pd.read_sql_query(query4, con=conn)
    opt_instore_delivery_item_mapping = pd.read_sql_query(query5, con=conn)
    
    
    merged_file = pd.merge(store_master, store_auto_premium_formula, on='gbl_store_id')
    user_profile['operator_name'] = user_profile['first_name'] + ' ' + user_profile['last_name']
    final_merge = pd.merge(merged_file, user_profile[['operator_name', 'user_id']], on='user_id')
    store_price_ending_rule = final_merge[['legacy_id', 'address1', 'operator_name', 'mcd_store_name', 'formula']]
    store_price_ending_rule = store_price_ending_rule.rename(columns={
        'legacy_id': 'store_id',
        'formula': 'price_ending_rule',
        'mcd_store_name': 'store_name',
        'address1': 'address',
    })
    store_price_ending_rule = clean_column_names(store_price_ending_rule)
    print("store_price_ending_rule")
    print(store_price_ending_rule.head())
    # load_table_to_db(store_price_ending_rule, config, 'store_price_ending_rule')
    
    item_grouping = opt_BaseItemGroups[['Item_ID', 'Item_Name', 'Final_ID', 'Item_Group_Name', 'Active']]

    item_groups = opt_BaseItemGroups.groupby('Final_ID')['Item_ID'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    item_grouping = pd.merge(item_grouping, item_groups.rename(columns={'Item_ID': 'items_in_rule'}), on='Final_ID')

    item_grouping = item_grouping.rename(columns={
        'Item_ID': 'item_id',
        'Item_Name': 'item_name',
        'Final_ID': 'final_id',
        'Item_Group_Name': 'item_group_name',
        'Active': 'active'
    })

    if not item_grouping.empty:
        item_grouping['round_id'] = round_id
        item_grouping = clean_column_names(item_grouping)
        # load_table_to_db(item_grouping, config, 'item_grouping')
    print("item_grouping")
    print(item_grouping.head())
    
    instore_delivery_item_mapping = opt_instore_delivery_item_mapping.rename(columns={
        '1': 'instore_item_id',
        '2': 'delivery_item_id'
    })
    if not instore_delivery_item_mapping.empty:
        instore_delivery_item_mapping['round_id'] = round_id
        instore_delivery_item_mapping = clean_column_names(instore_delivery_item_mapping)
        # load_table_to_db(instore_delivery_item_mapping, config, 'instore_delivery_item_mapping')
    print("instore_delivery_item_grouping") 
    print(instore_delivery_item_mapping.head())
    
    # create recommendation round table
    recommendation_round = pd.DataFrame(columns=['round_id', 'round_name'])
    recommendation_round.loc[len(recommendation_round)] = [round_id, round_name]
    # load_table_to_db(recommendation_round, config, 'recommendation_round')
    print("recommendation_round")
    print(recommendation_round)
    
    reports_to_load = config.load_tables.load_offline_reports.reports_to_load
    
    scenario_config_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.scenario_config}.xlsx"
    if 'scenario_config' in reports_to_load:
        scenario_config = s3.read(scenario_config_path)    
    scenario_config['BUSN_LCAT_ID_NU'] = scenario_config['BUSN_LCAT_ID_NU'].astype(str)
    scenario_constraints_merge = scenario_config.merge(store_master, left_on='BUSN_LCAT_ID_NU', right_on='gbl_store_id')
    
    scenario_constraints_merge_final = pd.merge(scenario_constraints_merge, user_profile[['operator_name', 'user_id']], on='user_id')
    scenario_constraints = scenario_constraints_merge_final[['legacy_id', 'mcd_store_name', 'address1', 'operator_name', 'Channel', 'max_pct_item_units_decrease', 'max_pct_store_units_decrease', 'max_pct_wap_increase', 'max_pct_wap_decrease', 'max_pct_gc_decrease', 'max_price_decrease', 'max_pct_gc_increase', 'max_pct_revenue_decrease', 'max_pct_margin_decrease', 'max_price_increase', 'max_pct_price_increase', 'max_pct_price_decrease', 'max_delivery_premium_decrease', 'max_delivery_premium_increase', 'max_flow_through_limit']]
    
    scenario_constraints = scenario_constraints.rename(columns={
        'legacy_id': 'store_id',
        'mcd_store_name': 'store_name',
        'address1': 'address',
        'Channel': 'channel',
        'max_pct_item_units_decrease': 'max_pct_item_units_decrease',
        'max_pct_store_units_decrease': 'max_pct_store_units_decrease',
        'max_pct_wap_increase': 'max_pct_wap_increase',
        'max_pct_wap_decrease': 'max_pct_wap_decrease',
        'max_pct_gc_decrease': 'max_pct_gc_decrease',
        'max_price_decrease': 'max_price_decrease',
        'max_pct_gc_increase': 'max_pct_gc_increase',
        'max_pct_revenue_decrease': 'max_pct_revenue_decrease',
        'max_pct_margin_decrease': 'max_pct_margin_decrease',
        'max_price_increase': 'max_price_increase',
        'max_pct_price_increase': 'max_pct_price_increase',
        'max_pct_price_decrease': 'max_pct_price_decrease',
        'max_delivery_premium_decrease': 'max_delivery_premium_decrease',
        'max_delivery_premium_increase': 'max_delivery_premium_increase',
        'max_flow_through_limit': 'max_flow_through_limit'
    })
    
    if not scenario_constraints.empty:
        scenario_constraints['round_id'] = round_id
        scenario_constraints = clean_column_names(scenario_constraints)
        # load_table_to_db(scenario_constraints, config, 'scenario_constraints')
    print("scenario_constraints")
    print(scenario_constraints.head())
    
    ladder_rules_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.ladder_rule_file}.xlsx"
    if 'ladder_rule_file' in reports_to_load:
        ladder_rules = s3.read(ladder_rules_path)
    ladder_rules = ladder_rules.rename(columns={
        'Ladder Group Name': 'ladder_group_name',
        'Item ID': 'item_id',
        'Item Name': 'item_name',
        'Tier Rank': 'tier_rank',
        'Can Equal Next Tier?': 'can_equal_next_tier'
    })
    item_groups = ladder_rules.groupby('ladder_group_name')['item_id'].apply(lambda x: [','.join(x.astype(str))]).reset_index()
    ladder_rules = pd.merge(ladder_rules, item_groups.rename(columns={'item_id': 'items_in_rule'}), on='ladder_group_name')
    
    rules_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.ladder_rule_with_size_file}.xlsx"
    if 'ladder_rule_with_size_file' in reports_to_load:
        ladder_rules_with_size = s3.read(rules_path)
    ladder_rules_with_size = ladder_rules_with_size.rename(columns={
        'Ladder Group Name': 'ladder_group_name',
        'Item ID': 'item_id',
        'Item Name': 'item_name',
        'Item Size': 'item_size',
        'Tier Rank': 'tier_rank',
        'Can Equal Next Tier?': 'can_equal_next_tier'
    })
    item_groups = ladder_rules_with_size.groupby('ladder_group_name')['item_id'].apply(lambda x: [','.join(x.astype(str))]).reset_index()
    ladder_rules_with_size = pd.merge(ladder_rules_with_size, item_groups.rename(columns={'item_id': 'items_in_rule'}), on='ladder_group_name')
    
    rules_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.soc_rule_file}.xlsx"
    if 'soc_rule_file' in reports_to_load:
        soc_rules = s3.read(rules_path)
    soc_rules = soc_rules.rename(columns={
        'EVM ID': 'evm_id',
        'EVM Name': 'evm_name',
        'Component Item ID': 'component_item_id',
        'Component Item Name': 'component_item_name',
        'Component Item Quantity': 'component_item_quantity',
        'Item type': 'item_type'
    })
    item_groups = soc_rules.groupby('evm_id')['component_item_id'].apply(lambda x: [','.join(x.astype(str))]).reset_index()
    soc_rules = pd.merge(soc_rules, item_groups.rename(columns={'component_item_id': 'items_in_rule'}), on='evm_id')
    
    rules_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.gap_rule_file}.xlsx"
    if 'gap_rule_file' in reports_to_load:
        price_gap_rules = s3.read(rules_path)
    price_gap_rules = price_gap_rules.rename(columns={
        'Item 1  ID': 'item_1_id',
        'Item 1 Name': 'item_1_name',
        'Item 2 ID': 'item_2_id',
        'Item 2 Name': 'item_2_name',
        'Minimum Price Gap': 'minimum_price_gap',
        'Maximum Price Gap': 'maximum_price_gap',
        'Constant Gap (Upcharge)': 'constant_gap_upcharge'
    })
    price_gap_rules['items_in_rule'] = price_gap_rules.apply(lambda row: [','.join([str(row['item_1_id']), str(row['item_2_id'])])], axis=1)
    
    if not ladder_rules.empty:
        ladder_rules['round_id'] = round_id
        ladder_rules = clean_column_names(ladder_rules)
        # load_table_to_db(ladder_rules, config, 'ladder_rules')
    print("ladder_rules")
    print(ladder_rules.head())
        
    if not ladder_rules_with_size.empty:
        ladder_rules_with_size['round_id'] = round_id
        ladder_rules_with_size = clean_column_names(ladder_rules_with_size)
        # load_table_to_db(ladder_rules_with_size, config, 'ladder_rules_with_size')
    print("ladder_rules_with_size")
    print(ladder_rules_with_size.head())
        
    if not soc_rules.empty:
        soc_rules['round_id'] = round_id
        soc_rules = clean_column_names(soc_rules)
        # load_table_to_db(soc_rules, config, 'soc_rules')
    print("soc_rules")
    print(soc_rules.head())
        
    if not price_gap_rules.empty:
        price_gap_rules['round_id'] = round_id
        price_gap_rules = clean_column_names(price_gap_rules)
        # load_table_to_db(price_gap_rules, config, 'price_gap_rules')
    print("gap_rules")
    print(price_gap_rules.head())
    
    exception_report_base_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.exception_report_at_base}.xlsx"
    if 'exception_report_at_base' in reports_to_load:
        exception_report_at_base_prices = s3.read(exception_report_base_path)
        
    # exception_report_recommend_path = f"{config.load_tables.path.parent}/{config.load_tables.market.name}/{config.load_tables.market.round_name}/{config.load_tables.path.exception_report_at_recommend}"
    # exception_report_at_recommend_prices = s3.read(exception_report_recommend_path, sheet_name='Store Summary')
    
    exception_report_at_base_prices['National Store Number'] = exception_report_at_base_prices['National Store Number'].astype(str)
    merged_files = exception_report_at_base_prices.merge(store_master, left_on='National Store Number', right_on='gbl_store_id')
    exception_report_summary_final_merge = pd.merge(merged_files, user_profile[['operator_name', 'user_id']], on='user_id')
    exception_report_summary = exception_report_summary_final_merge[['legacy_id', 'Channel', 'mcd_store_name', 'address1', 'operator_name', 'SOC - At Current Prices', 'SOC - At New Prices', 'Ladder - At Current Prices', 'Ladder - At New Prices', 'SPSP - At Current Prices', 'SPSP - At New Prices', 'Gap - At Current Prices', 'Gap - At New Prices', 'Cross - At Current Prices', 'Cross - At New Prices']]
    exception_report_summary = exception_report_summary.rename(columns={
        'legacy_id': 'store_id',
        'mcd_store_name': 'store_name',
        'address1': 'address',
        'Channel': 'channel',
        'SOC - At Current Prices': 'soc_violations_at_current_prices',
        'SOC - At New Prices': 'soc_violations_at_recommended_prices',
        'Ladder - At Current Prices': 'ladder_violations_at_current_prices',
        'Ladder - At New Prices': 'ladder_violations_at_recommended_prices',
        'SPSP - At Current Prices': 'spsp_violations_at_current_prices',
        'SPSP - At New Prices': 'spsp_violations_at_recommended_prices',
        'Gap - At Current Prices': 'gap_violations_at_current_prices',
        'Gap - At New Prices': 'gap_violations_at_recommended_prices',
        'Cross - At Current Prices': 'cross_violations_at_current_prices',
        'Cross - At New Prices': 'cross_violations_at_recommended_prices'
    })
    
    if not exception_report_summary.empty:
        exception_report_summary['round_id'] = round_id
        exception_report_summary = clean_column_names(exception_report_summary)
        # load_table_to_db(exception_report_summary, config, 'exception_report_summary')
    print("exception_report_summary")   
    print(exception_report_summary.head()) 
    
    s3 = S3(s3_bucket)
    # get the paths for the reports from the storage
    playbook_files = []
    if 'recommendation_playbook' in reports_to_load:
        playbook_files = s3.listdirs3(recommendation_playbook_path)
    
    # read recommendation playbook from the storage
    if len(playbook_files)>0:
        for file in playbook_files:
            df_modelled = s3.read(file, sheet_name=modelled_items_reason_code)
            df_modelled['STORE ID'] = df_modelled['STORE ID'].astype(str)
            store_master['id'] = store_master['id'].astype(str)
            df_modelled = df_modelled.merge(store_master[['id','gbl_store_id']],left_on='STORE ID',right_on='gbl_store_id',how='left')
            df_modelled.drop('gbl_store_id', axis=1, inplace=True)
            df_modelled['round_id'] = round_id
            df_modelled.columns = ['scenario', 'round_name', 'ownership', 'operator_name', 'store_name',
                'eotf', 'sssp_id', 'advisory', 'store_id', 'item_id',
                'item_name', 'channel', 'spsp', 'category', 'base_price',
                'recommended_price', 'recommended_price_change', 'instore_base_price',
                'instore_recommended_price', 'instore_price_change',
                'recommended_price_increase', 'adjustments_beyond_optimization',
                'base_sales', 'base_premium', 'new_premium', 'premium_change',
                'baseunits', 'margin_impact', 'margin_impact_percentage',
                'sales_impact', 'sales_impact_percentage',
                'incremented_optimized_price', 'impact_on_store_revenue',
                'impact_on_store_gc', 'selfelasticity_bin', 'price_bounds',
                'store_performance_units', 'store_performance_wap',
                'store_performance_gc', 'store_performance_delivery_premium',
                'store_performance_others', 'pricing_rules', 'pricing_rule_triggered',
                'reason_code', 'id', 'round_id'
            ]
            # load_table_to_db(df_modelled, config, 'recommendation_playbook_item_reason_code', playbook=True)
    
            df_non_modelled = s3.read(file, sheet_name=non_modelled_items_reason_code)
            df_non_modelled['STORE ID'] = df_non_modelled['STORE ID'].astype(str)
            df_non_modelled = df_non_modelled.merge(store_master[['id','gbl_store_id']],left_on='STORE ID',right_on='gbl_store_id',how='left')
            df_non_modelled.drop('gbl_store_id', axis=1, inplace=True)
            df_non_modelled['round_id'] = round_id
            df_non_modelled.columns = ['store_id', 'store_name', 'operator_name', 'ownership',
                'item_id', 'item_name', 'channel', 'category', 'spsp',
                'elasticity_flag', 'base_price', 'recommended_price', 'price_change',
                'ref_item_group_id', 'ref_item_group_name', 'post_processing_rule',
                'advisor', 'id', 'round_id'
            ]
            # load_table_to_db(df_non_modelled, config, 'recommendation_playbook_linked_item_reason_code', playbook=True)
            
# # load the config
# # config_path = '../../configs/data_files/data_path.yaml'
# config = config_manager
# if config.mcd.load_tables.load_offline_reports.run:
#     load_offline_reports(config)


user_config, data_config, model_config, debug_config = initialize_config()
# default_domain_name = "mcd"
# config.update_config(default_domain_name)
# config.initialize_config(default_domain_name)
config = data_config
print(config)
if config.load_tables.load_offline_reports.run:
    load_offline_reports(config)
