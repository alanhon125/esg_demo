from django.urls import path
from . import views

# TODO: update path
urlpatterns = [
    # extract or query doc parser output, response doc parser output
    path("generate_doc_parser_result", views.generate_doc_parser_result),
    # path("generate_table_extraction_result", views.generate_table_extraction_result),
    # to predict the string, location of metrics, numbers(values with unit) and relations(equal to/less than/greater than) from a sentence.
    path("metrics_predict", views.metrics_predict),
    # to predict the head entity(target), tail entity(method/guideline) and relations(reduced_by/increased_by/comply_with) from a sentence.
    path("reasoning_predict", views.reasoning_predict),
    path("generate_text_info", views.generate_text_info),

    path("save_text_info", views.save_text_info),

    path("generate_key_metrics", views.generate_key_metrics),

    # Import or update a schema from a csv in 'data/schema/', either target_metric.csv for metric extraction or target_aspect.csv for reasoning
    path("import_update_schema", views.import_update_schema),
    # extract/query doc parser output, then extract/query text metric entity relation, response doc parser output
    path("generate_text_entity_relation", views.generate_text_entity_relation),
    # (for debug) Update UIE extraction result JSON and database table name "metric_entity_relation" with latest relation, value, unit extraction function in uie_tools/utils.py
    path("update_text_entity_relation", views.update_text_entity_relation),

    # upload pdf file & save the basic information to postgresql database table name 'pdffiles'
    path("upload_file", views.upload_file),
    # get all extraction results for the processed pdf files
    path("get_uploadlist", views.get_uploadlist),
    # Single perform or extract document parsing, table metric extraction and metric entity-relation extraction, output all result as 'result.json' and to database table "metric_entity_relation"
    path("generate_all_results", views.generate_all_results),
    # get all compulsory or optinal intensity group metrics result
    path("get_compliance_checking_list", views.get_compliance_checking_list),
    # get all metric extraction results from table extraction for the processed pdf files
    path("view_metrics", views.view_metrics),
    # get all reasoning extraction results for the processed pdf files
    path("view_reasoning", views.view_reasoning),

    path("view_company_profile", views.view_company_profile),

    path("view_key_performance_kpi", views.view_key_performance_kpi),

    path("view_industry_average", views.view_industry_average),

    path("view_top_n_companies", views.view_top_n_companies),

    path("view_useful_metrics", views.view_useful_metrics),
    # get the options for subject, target_aspect and disclosure
    path("get_category_options", views.get_category_options),
    # (for debug) Insert table metric extraction result to database table "test_environment_info" with filename-page_id pairs provided
    path("extract_img_info", views.extract_img_info),
    # (for debug) Extract or perform document parsing with giving a list of filename keys
    path("extract_parsed_doc", views.extract_parsed_doc),
    # (for test) SCP files with given filenames (with .pdf), server IP address, username, password, port
    path("scp_files", views.scp_files),
    # (for test) Delete record of uploaded pdf with document_id on database table "pdffiles"
    path("delete_data", views.delete_data),
    # (for test) delete the files or results of particular tasks by giving filenames and task names
    path("delete_files",views.delete_files),
    # (for test) download the files or results of particular tasks by giving filenames and task names
    path("download_files",views.download_files),
    # (for test) Test UOM unification function and return multiplier and converted unit if success
    path("unify_uom", views.unify_uom),
    path("update_report_info",views.update_report_info),
    path("update_property_data",views.update_property_data),
    path('view_industries_average', views.view_industries_average),
    path('view_nested_metrics', views.view_nested_metrics),
    path('view_general_company_files_by_industries', views.view_general_company_files_by_industries),
    path('view_report_analysis', views.view_report_analysis),
    path('view_general_information_exposed_metrics', views.view_general_information_exposed_metrics),
    path('view_general_information_kpi_menu_by_company', views.view_general_information_kpi_menu_by_company),
    path('view_general_information_compliance_checking', views.view_general_information_compliance_checking),
    path('view_general_information_view_kpi_data_by_company_n_disclosure', views.view_general_information_view_kpi_data_by_company_n_disclosure),
    # Weight editing
    path('new_weighted_config', views.new_weighted_config),
    path('update_weighted_config_name', views.update_weighted_config_name),
    path('get_weighted_config', views.get_weighted_config),
    path('view_weighted_metrics', views.view_weighted_metrics),
    path('upsert_weighted_metrics', views.upsert_weighted_metrics),
]
