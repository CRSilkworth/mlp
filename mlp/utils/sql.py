
def query_with_kwargs(query_file_name, **kwargs):
  with open(query_file_name, 'r') as query_file:
    query = query_file.read()
    query = query.format(**kwargs)
  return query


def rolling_query(query_file_name, partition_field, order_field, num_leading, field_names, limit=None, **kwargs):
  select_str = '{field_name} AS {field_name}_0'
  lead_str = 'LEAD({field_name}, {num}) over (partition by {partition_field} order by {order_field} asc) as {field_name}_{num}'
  select_strs = []
  lead_strs = []
  for field_name in field_names:
    select_strs.append(select_str.format(field_name=field_name))
    for num in range(num_leading):
      lead_strs.append(lead_str.format(
        partition_field=partition_field,
        order_field=order_field,
        field_name=field_name,
        num=num+1
      ))

  select_columns = ',\n  '.join(select_strs + lead_strs)

  with open(query_file_name, 'r') as query_file:
    query = query_file.read()

  if limit is not None:
    limit_str = "LIMIT {}".format(limit)
  else:
    limit_str = ''

  query = query.format(
    select_columns=select_columns,
    limit=limit_str,
    **kwargs
  )
  return query
