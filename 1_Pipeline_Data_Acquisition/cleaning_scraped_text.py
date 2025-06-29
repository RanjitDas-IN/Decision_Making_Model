import re
import sys

# Path to the file
input_file_path = r'Temporary/TEMP.txt'
output_file_path = r"1_Pipeline_Data_Acquisition/Clean_data.txt"


# Read the file
with open(input_file_path, 'r', encoding='utf-8') as file:
    rawFile = file.read()

# IF HTML file extract body content else copy original
dataset = re.findall(r"<body>(.*?)</body>", rawFile, re.DOTALL)
if dataset:
    dataset = dataset[0]
    print("Body content extracted successfully!\n")
else:
    print("Not HTML, content unchanged!\n")
    dataset = rawFile


"""
Remove any unwanted html tags with its contents.

<(a|script) - matches the opening <a or <script> tag.

\b[^>]*?> - matches any attributes inside the tag until >.

.*? - non-greedy match for content inside the tag.

</\1> - matches the correct closing tag (</a> or </script>) using backreference \1.
"""
dataset = re.sub(r"<(a|script|h[0-9])\b[^>]*?>.*?</\1>", '', dataset, flags=re.DOTALL)
print("Remove unwanted html tags!\n")

"""
Remove Comments
"""
dataset = re.sub(r"<!--.*?-->", '', dataset, flags=re.DOTALL)
print("Removed html comments!\n")

"""
Remove multiple concutive new line
"""
dataset = re.sub(r'^\n +', "", dataset, flags=re.MULTILINE)
print("Remove cunsecutive new lines!\n")

"""
Remove multiple concutive spaces
"""
dataset = re.sub(r" +", " ", dataset)
print("RRemove multiple concutive spaces!\n")

"""
Remove empty lines
"""
dataset = re.sub(r"\n {1,}\n", "\n", dataset)
print("Remove empty lines!\n")

"""
Join Sentences empty lines
"""
dataset = re.sub(r'([^\.\?!])\n+', r'\1', dataset)
print("Join Sentences empty lines!\n")

"""
Remove lines that contain only content inside *double asterisks*
"""
dataset = re.sub(r"\*\*.+?\*\*", "", dataset)
print("Removed lines with double asterisks content!\n")

"""
Remove every char before before a newline
"""
dataset = re.sub(r"\*\*.+?\*\*", "", dataset)
print("Removed lines with double asterisks content!\n")
"""

Remove any special chrater from line start
"""
dataset = re.sub(r'^\-+', "", dataset, flags=re.MULTILINE)
print("Remove any special chrater from line start\n")

"""
Remove -- at line start
"""
dataset = re.sub(r"([^\w\s])\1+", r"\1", dataset)
print("Remove consecutive special characters\n")

# """
# Remove -- at line start
# """
# dataset = re.sub(r"\n", "|play\n", dataset)
# print("Remove -- at line start\n")

# """
# Append category for last element
# """
# dataset = dataset + "|play"
# print("Append category for last element\n")

"""
Remove starting space
"""
dataset = re.sub(r"^ +", "", dataset, flags=re.MULTILINE)
print("Remove starting space\n")


with open(output_file_path, 'w', encoding='utf-8') as out_file:
    out_file.write(dataset)