from django.db import models
from django.db.models import TextField, CharField, DateTimeField, JSONField, IntegerField, FloatField, BooleanField

class PdfFiles(models.Model):
    document_id = CharField(max_length=256, blank=True, null=True)
    filename = CharField(max_length=256, blank=True, null=True, db_index=True)
    industry = CharField(max_length=256, blank=True, null=True, db_index=True)
    company = CharField(max_length=256, blank=True, null=True, db_index=True)
    year = CharField(max_length=16, blank=True, null=True, db_index=True)
    report_language = CharField(
        max_length=256, blank=True, null=True, db_index=True)
    report_type = CharField(max_length=256, blank=True,
                            null=True, db_index=True)
    uploaded_date = DateTimeField(blank=True, null=True)
    process1_status = CharField(max_length=256, blank=True, null=True, db_index=True)
    process1_update_date = DateTimeField(blank=True, null=True)
    last_process1_elapsed_time = FloatField(null=True)
    process1_progress = FloatField(null=True)
    process2_status = CharField(max_length=256, blank=True, null=True, db_index=True)
    process2_update_date = DateTimeField(blank=True, null=True)
    last_process2_elapsed_time = FloatField(null=True)
    process2_progress = FloatField(null=True)
    last_process_date = DateTimeField(blank=True, null=True)
    status = CharField(max_length=256, blank=True, null=True, db_index=True)

    def __str__(self):
        return self.company

    class Meta:
        db_table = "pdffiles"
        ordering = ["-company"]

class PdfFilesInfo(models.Model):
    document_id = CharField(max_length=256, blank=True, null=True)
    filename = CharField(max_length=512, blank=True, null=True)
    stock_id = IntegerField(null=True)
    company_name = CharField(max_length=256, blank=True, null=True)
    report_year = IntegerField(null=True)
    report_type = CharField(max_length=256, blank=True, null=True)
    report_language = CharField(
        max_length=256, blank=True, null=True, db_index=True)
    page_count = IntegerField(null=True)
    filesize_mb = FloatField(null=True)
    exist_pdf = BooleanField(blank=True, null=True, db_index=True)
    exist_docParse = BooleanField(blank=True, null=True, db_index=True)
    exist_textMetric = BooleanField(blank=True, null=True, db_index=True)
    exist_textReasoning = BooleanField(blank=True, null=True, db_index=True)
    exist_tableMetric = BooleanField(blank=True, null=True, db_index=True)

    def __str__(self):
        return self.company_name

    class Meta:
        db_table = "pdffiles_info"
        ordering = ["-company_name"]

class CompanyIndustry(models.Model):

    stock_id = IntegerField(null=True)
    company_name_ch = CharField(max_length=256, blank=True, null=True)
    company_name_en = CharField(max_length=256, blank=True, null=True)
    industry = CharField(max_length=256, blank=True, null=True)
    sector = CharField(max_length=256, blank=True, null=True)
    subsector = CharField(max_length=256, blank=True, null=True)

    def __str__(self):
        return self.stock_id

    class Meta:
        db_table = "company_industry"
        ordering = ["-stock_id"]