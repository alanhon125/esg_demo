from django.db import models
from django.db.models import TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField, DateTimeField

class MetricEntityRelationUnfiltered(models.Model):
    document_id = CharField(max_length=32, blank=True, null=True)
    company_name = CharField(max_length=512, blank=True, null=True, db_index=True)
    year = CharField(max_length=16, blank=True, null=True, db_index=True)
    page_id = IntegerField(blank=True, null=True, db_index=True)
    text_block_id = IntegerField(blank=True, null=True, db_index=True)
    block_element = CharField(max_length=512, blank=True, null=True, db_index=True)
    sent_id = IntegerField(blank=True, null=True, db_index=True)
    sentence = TextField(blank=True, null=True, db_index=True)
    isMatchedKeyword = BooleanField(blank=True, null=True, db_index=True)
    metric_year = CharField(max_length=16, blank=True, null=True, db_index=True)
    metric = CharField(max_length=512, blank=True, null=True, db_index=True)
    metric_char_position = CharField(max_length=512, blank=True, null=True, db_index=True)
    subject = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_aspect = CharField(max_length=512, blank=True, null=True, db_index=True)
    disclosure = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_metric = CharField(max_length=512, blank=True, null=True, db_index=True)
    compulsory = BooleanField(blank=True, null=True, db_index=True)
    intensity_group = IntegerField(blank=True, null=True, db_index=True)
    similarity = FloatField(max_length=16, blank=True, null=True, db_index=True)
    relation = CharField(max_length=512, blank=True, null=True, db_index=True)
    number = CharField(max_length=512, blank=True, null=True, db_index=True)
    number_char_position = CharField(max_length=512, blank=True, null=True, db_index=True)
    original_value = CharField(max_length=512, blank=True, null=True, db_index=True)
    unit = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_unit = CharField(max_length=512, blank=True, null=True, db_index=True)
    uom_conversion_multiplier = FloatField(blank=True, null=True, db_index=True)
    converted_value = CharField(max_length=512, blank=True, null=True, db_index=True)
    converted_unit = CharField(max_length=512, blank=True, null=True, db_index=True)
    update_datetime = DateTimeField(blank=True, null=True, db_index=True)

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "metric_entity_relation_unfiltered"
        ordering = ["-company_name","-year"]

class MetricSchema(models.Model):
    subject_no = CharField(max_length=4, blank=True, null=True, db_index=True)
    subject = CharField(max_length=32, blank=True, null=True, db_index=True)
    aspect_no = CharField(max_length=4, blank=True, null=True, db_index=True)
    target_aspect = CharField(max_length=512, blank=True, null=True, db_index=True)
    disclosure_no = CharField(max_length=4, blank=True, null=True, db_index=True)
    disclosure = CharField(max_length=512, blank=True, null=True, db_index=True)
    metric = CharField(max_length=512, blank=True, null=True, db_index=True)
    category = CharField(max_length=512, blank=True, null=True, db_index=True)
    compulsory = BooleanField(blank=False, null=False, db_index=True)
    intensity_group = IntegerField(blank=True, null=True, db_index=True)
    esg_field_for_astri_ref = BooleanField(blank=False, null=False, db_index=True)
    definition = CharField(max_length=512, blank=True, null=True, db_index=True)
    unit = CharField(max_length=512, blank=True, null=True, db_index=True)
    density_NTP_kg_per_m3 = FloatField(max_length=16, blank=True, null=True, db_index=True)
    data_type = CharField(max_length=512, blank=True, null=True, db_index=True)
    source = CharField(max_length=512, blank=True, null=True, db_index=True)

    def __str__(self):
        return self.metric

    class Meta:
        db_table = "metric_schema"
        ordering = ["-metric"]