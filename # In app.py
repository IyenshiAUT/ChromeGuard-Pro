# In app.py, in extract_endpoint
file = request.files['image']
print('Extract file size:', len(file.read()))
file.seek(0)  # Reset file pointer after reading