def pretty_print(d: dict) -> None:
	margin = len(max(d, key=len))
	for k in d:
		print(f'{k:{margin}} : {d[k]:.5f}')
	print()

