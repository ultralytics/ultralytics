from ultralytics.data.explorer.explorer import Explorer

exp = Explorer()
exp.create_embeddings_table()

print(exp.label_count(["person", "dog"]))
