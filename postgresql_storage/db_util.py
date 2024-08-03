from django.apps import apps

class DbUtil:
    def __init__(self):
        model_dict = {}
        model_field_dict = {}
        for model in apps.get_models():
            model_dict[model._meta.db_table] = model
            model_field_dict[model._meta.db_table] = [f.name for f in model._meta.fields]

        self.model_dict = model_dict
        self.model_field_dict = model_field_dict


    def __iter__(self):
        return list(self.model_dict.items())


    def get_all_tables(self):
        """
        return all tables on db.
        """
        return list(self.model_dict.keys())


    def get_model_instance(self, table_name):
        """
        Get table object through table_name for more complex operation.
        :param table_name:
        :return:
        """
        if table_name not in self.model_dict.keys():
            raise Exception("No such table {} in database".format(table_name))
        return self.model_dict[table_name]


    def select_table(
            self,
            table_name,
            field_list=[],
            filter_dict=None,
            filter_Q=None
    ):
        """
        select method, see demos in postgersql_storage readme.md
        :param table_name:
        :param field_list:
        :param filter_dict:
        :param filter_Q:
        :return: query result list
        """
        if table_name not in self.model_dict.keys():
            raise Exception("No such table {} in database".format(table_name))

        model_object = self.model_dict[table_name]

        wrong_field_list = set(field_list) - set(self.model_field_dict[table_name])
        if len(wrong_field_list) > 0:
            raise Exception("Fields: \"{}\" are not existed on table {}, \nPlease use those fields: \"{}\"".format(
                " | ".join(wrong_field_list),
                table_name,
                " | ".join(self.model_field_dict[table_name])
            ))

        if not filter_dict and not filter_Q:
            result = model_object.objects.all().values(*field_list)

        elif filter_dict and not filter_Q:
            result = model_object.objects.filter(**filter_dict).values(*field_list)

        elif not filter_dict and filter_Q:
            result = model_object.objects.filter(filter_Q).values(*field_list)

        else:
            result = model_object.objects.filter(filter_Q, **filter_dict).values(*field_list)

        return list(result)


    def insert_data(self, table_name, data_list):
        if table_name not in self.model_dict.keys():
            raise Exception("No such table {} in database".format(table_name))

        for i, data in enumerate(data_list):
            wrong_field_list = set(data.keys()) - set(self.model_field_dict[table_name])
            if len(wrong_field_list) > 0:
                raise Exception("Fields: \"{}\" are not existed on table {}, please check index: {}".format(
                    " | ".join(wrong_field_list),
                    table_name,
                    i
                ))

        model_object = self.model_dict[table_name]
        create_list = []

        for data in data_list:
            create_list.append(
                model_object(**data)
            )

        model_object.objects.bulk_create(create_list,ignore_conflicts=True)


    def update_data(
            self,
            table_name,
            filter_dict=None,
            update_dict=None
    ):
        if table_name not in self.model_dict.keys():
            raise Exception("No such table {} in database".format(table_name))

        model_object = self.model_dict[table_name]

        if filter_dict and update_dict:
            model_object.objects.filter(**filter_dict).update(**update_dict)


    def delete_data(
            self,
            table_name,
            filter_dict=None
    ):
        if table_name not in self.model_dict.keys():
            raise Exception("No such table {} in database".format(table_name))

        model_object = self.model_dict[table_name]

        if filter_dict:
            model_object.objects.filter(**filter_dict).delete()
