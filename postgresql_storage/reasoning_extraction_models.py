from django.db import models
from django.db.models import TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

class ReasoningEntityRelationUnfiltered(models.Model):
    document_id = CharField(max_length=32, blank=True, null=True)
    company_name = CharField(max_length=512, blank=True, null=True, db_index=True)
    year = CharField(max_length=16, blank=True, null=True, db_index=True)
    page_id = IntegerField(blank=True, null=True, db_index=True)
    text_block_id = IntegerField(blank=True, null=True, db_index=True)
    block_element = CharField(max_length=512, blank=True, null=True, db_index=True)
    sent_id = IntegerField(blank=True, null=True, db_index=True)
    sentence = TextField(blank=True, null=True, db_index=True)
    isMatchedKeyword = BooleanField(blank=True, null=True, db_index=True)
    head_entity_type = CharField(max_length=512, blank=True, null=True, db_index=True)
    head_entity = CharField(max_length=512, blank=True, null=True, db_index=True)
    head_entity_char_position = CharField(max_length=512, blank=True, null=True, db_index=True)
    subject = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_aspect = CharField(max_length=512, blank=True, null=True, db_index=True)
    disclosure = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_metric = CharField(max_length=512, blank=True, null=True, db_index=True)
    similarity = FloatField(max_length=16, blank=True, null=True, db_index=True)
    relation = CharField(max_length=512, blank=True, null=True, db_index=True)
    tail_entity_type = CharField(max_length=512, blank=True, null=True, db_index=True)
    tail_entity = CharField(max_length=512, blank=True, null=True, db_index=True)
    tail_entity_char_position = CharField(max_length=512, blank=True, null=True, db_index=True)
    update_datetime = DateTimeField(blank=True, null=True, db_index=True)

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "reasoning_entity_relation_unfiltered"
        ordering = ["-company_name","-year"]

class TargetAspect(models.Model):
    aspect_id = CharField(max_length=32, blank=True, null=True)
    subject = CharField(max_length=512, blank=True, null=True, db_index=True)
    target_aspect = CharField(max_length=512, blank=True, null=True, db_index=True)
    keywords = CharField(max_length=512, blank=True, null=True, db_index=True)

    def __str__(self):
        return self.target_aspect

    class Meta:
        db_table = "target_aspect"
        ordering = ["-aspect_id"]