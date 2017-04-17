import json
import urllib.request
import html
import re

with open('data/all_guides.json') as json_data:
    d = json.load(json_data)

# print(d)
print(len(d))

for guide in d:
    print(guide["id"], guide["name"], len(guide["pages"]))
    for page in guide["pages"]:
        if len(page["redirect_url"]):
            break
        print(" - ", page["id"], page["name"])

        url = "http://guides.temple.edu/c.php?g=%s&p=%s" % (guide["id"], page["id"])
        print("   - ", url)
        regex = re.compile('[^a-zA-Z]')
        filename = regex.sub('', page["name"]) + "_" + guide["id"] + "_" + page["id"]
        filepath = "data/guides/" + filename + ".json"
        print("   - ", filename)
        print("   - ", filepath)
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        # f = open(filepath, "wb")
        # f.write(content)
        # f.close()
        import json

        with open(filepath, 'w') as outfile:
            json.dump([guide["id"], page["id"], content], outfile)

