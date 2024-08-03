from django.urls import path, include
from . import views

# TODO: update path
urlpatterns = [
    # extract or query doc parser output, response doc parser output
    path("genius_inference/", views.genius_inference), 
    # path("generate_table_extraction_result", views.generate_table_extraction_result),
]