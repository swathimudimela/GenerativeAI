### Integrate HTML with flask : Also called JINJA technique
## To read posted values we need one more library : request

from flask import Flask, redirect , url_for , render_template , request

app = Flask(__name__)

@app.route("/")
def welcome():
    # render a html page
    return render_template('index.html')

@app.route('/final_result/<int:score>')
def final_result(score):
    res = ""
    if score >= 50:
        res = "PASS"
    else:
        res = "Fail"
    return render_template('result.html',result = res)



## Check results using HTML page
@app.route('/submit',methods = ['POST','GET'])
def submit():
    total_score = 0
    print("Iam here")
    if request.method == 'POST' :
        science = float(request.form['science'])
        print("Got ",science)
        maths = float(request.form['maths'])
        c = float(request.form['c'])
        ds = float(request.form['DS'])
        total_score = (science+maths+c+ds)/4
    res = "final_result"
    print(" Total Score we got is ", total_score)
    
    return redirect(url_for(res,score = total_score))
    



if __name__ == '__main__' :
    app.run(debug = True)
