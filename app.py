from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def health_recommendation():
    if request.method == 'POST':
        name = request.form['name']
        contact = request.form['contact']
        address = request.form['address']
        gender = request.form['gender']
        age = request.form['age']
        health_note = request.form['health_note']
    
        recommendation = "Get sufficient rest and consult a healthcare professional if needed."

        return render_template('result.html', name=name, contact=contact, address=address,
                               gender=gender, age=age, health_note=health_note,
                               recommendation=recommendation)

    return render_template('form.html')

    

if __name__ == '__main__':
    app.run(debug=True)
