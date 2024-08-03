## DbUtil usage document

Create container
```shell
docker-compose -f postgresql_compose.yml up -d
```
### 1. Create new table

Create your table on model file under **postgresql_storage** folder, and apply migration.

And make you've added **db_table** in Meta class.

**Before run CRUD function, make sure your model has been created on DB. (use django migrate)**

Django Model reference: https://docs.djangoproject.com/zh-hans/3.2/topics/db/models/

```python
class FlashReportByParcel(models.Model):
    doc_id = AutoField(primary_key=True)
    doc_name = CharField(max_length=32, blank=True, null=True)
    date_of_offence = DateField()
    place_of_offence = CharField(max_length=32, blank=True, null=True)
    fight_no = CharField(max_length=32, blank=True, null=True)
    route = CharField(max_length=32, blank=True, null=True)
    goods_declaration = CharField(max_length=32, blank=True, null=True)
    consigment_quantity = CharField(max_length=32, blank=True, null=True)
    consigment_packing = CharField(max_length=32, blank=True, null=True)
    goods = CharField(max_length=32, blank=True, null=True)
    quantity = CharField(max_length=32, blank=True, null=True)
    value = CharField(max_length=32, blank=True, null=True)
    sender = CharField(max_length=32, blank=True, null=True)
    recipient = CharField(max_length=32, blank=True, null=True)
    concealment_method = CharField(max_length=32, blank=True, null=True)
    seizure_package = CharField(max_length=32, blank=True, null=True)
    outline = CharField(max_lengtmh=32, blank=True, null=True)

    class Meta:
        db_table = "flash_report_by_parcel"
```
```shell
python manage.py makemigrations
python manage.py migrate
```


### 2. Show all tables on DB

It will return all tables that have been migrated to DB from your model file
```python
from postgresql_storage.db_util import DbUtil
du = DbUtil()
table_list = du.get_all_tables()
print(table_list)
```
```
['django_admin_log', 'auth_permission', 'auth_group', 'auth_user', 'django_content_type', 'django_session', 'ner_news', 'ner_news_entity', 'entity_definition', 'pdf_type_keyword', 'pdf_field_mapping', 'quarterly_review', 'flash_report_by_parcel', 'flash_report_by_passenger', 'press_release']
```

### 3. Insert data to DB

Function `insert_data` on DbUtil is designed for inserting data to table.

You need provide table_name and data_list in this function, and `insert_data` support bulk insert, just input your data as a dict format on data_list.

If table_name is not existed on DB and table field you provided is not existed, if will raise error.

Fields:

- **table_name**: Insert to which table.
- **data_list**: data_list need to insert, each element in the list should be dict.
```python
from postgersql_storage.db_util import DbUtil
du = DbUtil()

"""
INSERT into flash_report_by_parcel (doc_name, date_of_offence, place_of_offence, fight_no, route, goods)
VALUES 
('example.pdf', '2021-12-07', 'HK', 'AB1234', NULL, 'meth'),
('123.pdf', '2019-12-20', 'China', 'QS1234', 'HK to CHN', 'cocaine'),
('ppp.pdf', '2020-12-04', 'Macao', 'BB0987', NULL, 'weed'),
('567.pdf', '2021-02-01', 'Hong Kong', 'YU4567', 'JP to HK', 'cannabis');
"""

data_list = [
    {
        "doc_name": "example.pdf",
        "date_of_offence": "2021-12-07",
        "place_of_offence": "HK",
        "fight_no": "AB1234",
        "route": None,
        "goods": "meth"
    },
    {
        "doc_name": "123.pdf",
        "date_of_offence": "2019-12-20",
        "place_of_offence": "China",
        "fight_no": "QS1234",
        "route": "HK to CHN",
        "goods": "cocaine"
    },
    {
        "doc_name": "ppp.pdf",
        "date_of_offence": "2020-12-04",
        "place_of_offence": "Macao",
        "fight_no": "BB0987",
        "route": None,
        "goods": "weed"
    },
    {
        "doc_name": "567.pdf",
        "date_of_offence": "2021-02-01",
        "place_of_offence": "Hong Kong",
        "fight_no": "YU4567",
        "route": 'JP to HK',
        "goods": "cannabis"
    }
]
# save to table "flash_report_by_parcel"
du.insert_data("flash_report_by_parcel", data_list=data_list)
```

### 4. Query function:
Function `select_table` on DbUtil is designed for selecting data from table.

Fields:

- **table_name**: str, which table need select
- **field_list**: list, input fields that need to be output. If input a empty list, it will return all fields.
- **filter_dict**: dict, default is None. select condition, will be passed to django model filter parameters. 
  Every condition in filter_dict uses **AND** to connect, if you want to represent **OR** relation, please user filter_Q.
- **filter_Q**: django.db.models.Q object, used for complex query. 
  Default is None.

Django filter reference: https://docs.djangoproject.com/zh-hans/3.2/topics/db/queries/

Django Q operation reference: https://docs.djangoproject.com/zh-hans/3.2/topics/db/queries/#complex-lookups-with-q-objects

```python
from postgersql_storage.db_util import DbUtil
du = DbUtil()

"""
SELECT * FROM flash_report_by_parcel;
"""
result1 = du.select_table(table_name="flash_report_by_parcel", field_list=[]) # leave filed_list a empty list to get all fields.
print("select all: ")
print(result1)

"""
SELECT doc_name, goods FROM flash_report_by_parcel 
WHERE doc_name='567.pdf';
"""
result2 = du.select_table(
    table_name="flash_report_by_parcel",
    field_list=["doc_name", "goods"],
    filter_dict={
        "doc_name": "567.pdf"
    }
)
print("exact select: ")
print(result2)

"""
SELECT doc_name, date_of_offence, place_of_offence, fight_no, route, goods FROM flash_report_by_parcel 
WHERE (date_of_offence >= '2020-01-01' AND date_of_offence <= '2021-01-01');
"""
result3 = du.select_table(
    table_name="flash_report_by_parcel",
    field_list=["doc_name", "date_of_offence", "place_of_offence", "fight_no", "route", "goods"],
    filter_dict={
        "date_of_offence__gte": "2020-01-01",
        "date_of_offence__lte": "2021-01-01"
    }
)
print("compare select: ")
print(result3)

"""
SELECT doc_name, date_of_offence, place_of_offence, fight_no, route, goods FROM flash_report_by_parcel 
WHERE (date_of_offence >= '2020-01-01')
AND (place_of_offence CONTAINS 'HK' OR place_of_offence CONTAINS 'Hong Kong')
AND (goods = 'cannabis' OR 'marijuana' OR 'weed');
"""
from django.db.models import Q
result4 = du.select_table(
    table_name="flash_report_by_parcel",
    field_list=["doc_name", "date_of_offence", "place_of_offence", "fight_no", "route", "goods"],
    filter_dict = {
        "date_of_offence__gte": "2020-01-01",
        "goods__in": ["cannabis", "marijuana", "weed", "meth"]
    },
    filter_Q = Q(place_of_offence__icontains="HK") | Q(place_of_offence__icontains="Hong Kong")
)
print("complex select: ")
print(result4)

### try contain list ###
"""
SELECT doc_name, date_of_offence, place_of_offence, fight_no, route, goods FROM flash_report_by_parcel 
WHERE (date_of_offence >= '2018-01-01' AND date_of_offence < '2022-01-01')
AND (place_of_offence CONTAINS 'HK' OR place_of_offence CONTAINS 'Hong Kong' OR place_of_offence CONTAINS 'China');
"""
from django.db.models import Q
contain_list = ["HK", "Hong Kong", "China"]
filter_Q = Q()
for contain in contain_list:
    filter_Q = Q(place_of_offence__icontains=contain) | filter_Q

result5 = du.select_table(
    table_name="flash_report_by_parcel",
    field_list=["doc_name", "date_of_offence", "place_of_offence", "fight_no", "route", "goods"],
    filter_dict={
        "date_of_offence__gte": "2018-01-01",
        "date_of_offence__lt": "2022-01-01"
    },
    filter_Q=filter_Q
)

print("contain list select: ")
print(result5)
```

```
select all: 
[{'doc_id': 1, 'doc_name': 'example.pdf', 'date_of_offence': datetime.date(2021, 12, 7), 'place_of_offence': 'HK', 'fight_no': 'AB1234', 'route': None, 'goods_declaration': None, 'consigment_quantity': None, 'consigment_packing': None, 'goods': 'meth', 'quantity': None, 'value': None, 'sender': None, 'recipient': None, 'concealment_method': None, 'seizure_package': None, 'outline': None}, {'doc_id': 2, 'doc_name': '123.pdf', 'date_of_offence': datetime.date(2019, 12, 20), 'place_of_offence': 'China', 'fight_no': 'QS1234', 'route': 'HK to CHN', 'goods_declaration': None, 'consigment_quantity': None, 'consigment_packing': None, 'goods': 'cocaine', 'quantity': None, 'value': None, 'sender': None, 'recipient': None, 'concealment_method': None, 'seizure_package': None, 'outline': None}, {'doc_id': 3, 'doc_name': 'ppp.pdf', 'date_of_offence': datetime.date(2020, 12, 4), 'place_of_offence': 'Macao', 'fight_no': 'BB0987', 'route': None, 'goods_declaration': None, 'consigment_quantity': None, 'consigment_packing': None, 'goods': 'weed', 'quantity': None, 'value': None, 'sender': None, 'recipient': None, 'concealment_method': None, 'seizure_package': None, 'outline': None}, {'doc_id': 4, 'doc_name': '567.pdf', 'date_of_offence': datetime.date(2021, 2, 1), 'place_of_offence': 'Hong Kong', 'fight_no': 'YU4567', 'route': 'JP to HK', 'goods_declaration': None, 'consigment_quantity': None, 'consigment_packing': None, 'goods': 'cannabis', 'quantity': None, 'value': None, 'sender': None, 'recipient': None, 'concealment_method': None, 'seizure_package': None, 'outline': None}]
exact select: 
[{'doc_name': '567.pdf', 'goods': 'cannabis'}]
compare select: 
[{'doc_name': 'ppp.pdf', 'date_of_offence': datetime.date(2020, 12, 4), 'place_of_offence': 'Macao', 'fight_no': 'BB0987', 'route': None, 'goods': 'weed'}]
complex select: 
[{'doc_name': 'example.pdf', 'date_of_offence': datetime.date(2021, 12, 7), 'place_of_offence': 'HK', 'fight_no': 'AB1234', 'route': None, 'goods': 'meth'}, {'doc_name': '567.pdf', 'date_of_offence': datetime.date(2021, 2, 1), 'place_of_offence': 'Hong Kong', 'fight_no': 'YU4567', 'route': 'JP to HK', 'goods': 'cannabis'}]
contain list select: 
[{'doc_name': 'example.pdf', 'date_of_offence': datetime.date(2021, 12, 7), 'place_of_offence': 'HK', 'fight_no': 'AB1234', 'route': None, 'goods': 'meth'}, {'doc_name': '123.pdf', 'date_of_offence': datetime.date(2019, 12, 20), 'place_of_offence': 'China', 'fight_no': 'QS1234', 'route': 'HK to CHN', 'goods': 'cocaine'}, {'doc_name': '567.pdf', 'date_of_offence': datetime.date(2021, 2, 1), 'place_of_offence': 'Hong Kong', 'fight_no': 'YU4567', 'route': 'JP to HK', 'goods': 'cannabis'}]
```

### 5. Update function
Fields:
- **table_name**: Update on which table.
- **filter_dict**: filter condition, use django model filter dict format.
- **update_dict**: update value, use dict format.

```python
from postgersql_storage.db_util import DbUtil
du = DbUtil()

"""
UPDATE flash_report_by_parcel
SET doc_name = 'new_doc.pdf', place_of_offence = 'Hong Kong'
WHERE doc_id = 1;
"""
filter_dict = {
    "doc_id": 1
}

update_dict = {
    "doc_name": "new_doc.pdf",
    "place_of_offence": "Hong Kong"
}

du.update_data("flash_report_by_parcel", filter_dict=filter_dict, update_dict=update_dict)
```

### 6. Delete function
Fields
- **table_name**: In which table to delete.
- **delete_dict**: filter operation.

```python
from postgersql_storage.db_util import DbUtil
du = DbUtil()
"""
DELETE FROM flash_report_by_parcel
WHERE doc_id = 1;
"""
delete_dict = {
    "doc_id": 1
}
du.delete_data("flash_report_by_parcel", delete_dict)

"""
DELETE FROM pdf_field_mapping
WHERE pdf_type = 'flash_report' AND pdf_field = 'CUSTOMS' AND table_field = 'customs';
"""
delete_dict = {
    "pdf_type": 'flash_report',
    "pdf_field": 'CUSTOMS',
    "table_field": 'customs'
}
du.delete_data("pdf_field_mapping", delete_dict)
```