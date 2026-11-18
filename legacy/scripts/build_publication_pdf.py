import typst
import shutil

with open("paper/paper.typ", encoding="utf-8") as f:
    text = f.read()

# Add a pagebreak right before Algorithm 1 so it starts at the top of page 7
text = text.replace(
    "#v(0.5em)\n#align(center)[\n#block(\n  fill: rgb(\"fafafa\"),\n  stroke: 0.6pt + rgb(\"444444\"),",
    "#pagebreak()\n#align(center)[\n#block(\n  fill: rgb(\"fafafa\"),\n  stroke: 0.6pt + rgb(\"444444\"),"
)

with open("paper/paper.typ", "w", encoding="utf-8") as f:
    f.write(text)

typst.compile("paper/paper.typ", output="paper/despin_paper.pdf")
print("Compiled paper/despin_paper.pdf successfully!")
shutil.copyfile("paper/despin_paper.pdf", "despin_paper.pdf")
print("Copied to ./despin_paper.pdf successfully!")
