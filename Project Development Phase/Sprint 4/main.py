from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
cmodel = pickle.load(open('resale_model.pkl', 'rb'))
autos = pd.read_csv('car_resale_preprocessed.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/c_predict', methods=['POST'])
def c_predict():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    regyear = int(request.form['reg_year'])
    powerps = float(request.form['car_power'])
    kms = float(request.form['kilo_driven'])
    regmonth = int(months.index(request.form.get('reg_month')))+1
    gearbox = request.form['gear_type']
    damage = request.form['car_condition']
    model = request.form.get('model')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuel_type')
    vehicletype = request.form.get('veh_type')
    new_row = {'yearOfRegistration': regyear,
               'monthOfRegistration': regmonth,
               'gearbox': gearbox, 'notRepairedDamage': damage,
               'model': model, 'brand': brand, 'fuelType': fuelType,
               'vehicleType': vehicletype, 'powerPS': powerps, 'kilometer': kms}
    print(new_row)
    new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox',
                                   'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuel Type',
                                   'brand', 'notRepairedDamage'])
    new_df = new_df.append(new_row, ignore_index=True)
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes' + i + '.npy'),allow_pickle=True)
        val = int(np.where(mapper[i].classes_ == new_row[i])[0][0])
        print(i, new_row[i], val)
        new_df.loc[:, i + '_labels'] = val
    labeled = new_df[['yearOfRegistration','powerPS'
            , 'kilometer'
            , 'monthOfRegistration']
         + [x + '_labels' for x in labels]]
    X = labeled.values
    print(X)
    y_prediction = cmodel.predict(X)
    print(y_prediction)
    return 'The resale value predicted is {:.2f}$'.format(y_prediction[0])


@app.route('/car_price', methods=['GET', 'POST'])
def car_price():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    fuel_types=autos['fuelType'].unique()
    brands = autos['brand'].unique()
    models = autos['model'].unique()
    vehicle_types = autos['vehicleType'].unique()
    return render_template('carPrice.html', fuel_types=fuel_types, months=months, brands=brands, models=models, vehicle_types=vehicle_types)


if __name__ == '__main__':
    app.run(debug=True)
