from flask import Flask, redirect, url_for, render_template, request
import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
#from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from matplotlib import pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
		return render_template("index.html")


@app.route("/requestGraph", methods= ["POST", "GET"])
def requestGraph():
		if request.method == "POST":
				graphType = request.form["requestGraph"]
				return redirect(url_for("graph", graphType=graphType))
		else:
				return render_template("requestGraphButton.html")

@app.route("/<graphType>")
def graph(graphType):
		if graphType == "pie":
			return redirect(url_for("pieGraph"))
		else:
			return redirect(url_for("barGraph"))


# @app.route("/pie")
# def pieGraph():
# 	fig = plt.figure()
# 	ax = fig.add_axes([0,0,1,1])
# 	ax.axis('equal')
# 	langs = ['C', 'C++', 'Java', 'Python', 'PHP']
# 	students = [23,17,35,29,12]
# 	ax.pie(students, labels = langs,autopct='%1.2f%%')
# 	script, div = components(fig)
# 	return render_template("pie.html", 
#                            the_div=div, the_script=script)

			
		


if __name__ == "__main__":
    app.run(debug=True)