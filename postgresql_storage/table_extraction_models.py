from django.db import models
from django.db.models import TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField

class EnvironmentInfo(models.Model):
    document_id = CharField(max_length=32, blank=True, null=True)
    company_name = CharField(max_length=256, blank=True, null=True, db_index=True)
    position = CharField(max_length=256, blank=True, null=True, db_index=True)
    metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    table_id = CharField(max_length=32, blank=True, null=True)
    table_area = CharField(max_length=256, blank=True, null=True, db_index=True)
    target_metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    converted_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    year = CharField(max_length=256, blank=True, null=True, db_index=True)
    value = CharField(max_length=256, blank=True, null=True, db_index=True)
    converted_value = FloatField(null=True)
    similar_score = FloatField()
    page_no = IntegerField(null=True)
    raw_value = CharField(max_length=256, blank=True, null=True, db_index=True)
    raw_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    new_metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    label = CharField(max_length=128, blank=True, null=True, db_index=True)
    multiplier = FloatField(null=True)
    converted_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    table_score = FloatField()

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "environment_info"
        ordering = ["-company_name"]


class TestEnvironmentInfo(models.Model):
    document_id = CharField(max_length=32, blank=True, null=True)
    company_name = CharField(max_length=256, blank=True, null=True, db_index=True)
    position = CharField(max_length=256, blank=True, null=True, db_index=True)
    metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    table_id = CharField(max_length=32, blank=True, null=True)
    table_area = CharField(max_length=256, blank=True, null=True, db_index=True)
    target_metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    converted_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    year = CharField(max_length=256, blank=True, null=True, db_index=True)
    value = CharField(max_length=256, blank=True, null=True, db_index=True)
    converted_value = FloatField(null=True)
    similar_score = FloatField()
    page_no = IntegerField(null=True)
    raw_value = CharField(max_length=256, blank=True, null=True, db_index=True)
    raw_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    new_metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    label = CharField(max_length=128, blank=True, null=True, db_index=True)
    multiplier = FloatField()
    converted_unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    table_score = FloatField()

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "test_environment_info"
        ordering = ["-company_name"]


class TestTextInfo(models.Model):
    company_name = CharField(max_length=256, blank=True, null=True, db_index=True)
    # target_metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    metric = CharField(max_length=256, blank=True, null=True, db_index=True)
    unit = CharField(max_length=256, blank=True, null=True, db_index=True)
    year = CharField(max_length=16, blank=True, null=True, db_index=True)
    value = CharField(max_length=256, blank=True, null=True, db_index=True)
    # similar_score = FloatField()

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "test_text_info"
        ordering = ["-company_name"]
