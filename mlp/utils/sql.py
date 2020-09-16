
def query_with_kwargs(query_file_name, **kwargs):
  with open(query_file_name, 'r') as query_file:
    query = query_file.read()
    query = query.format(**kwargs)
  return query
