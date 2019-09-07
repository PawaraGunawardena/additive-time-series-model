import pandas as pd
from fbprophet import Prophet

def additive_model(train_data, test_data, _change_point_prior_scale, _yearly_seasonality):
    prophet = Prophet(changepoint_prior_scale=_change_point_prior_scale,
                      yearly_seasonality=_yearly_seasonality)

    prophet.fit(train_data)

    future_frame = prophet.make_future_dataframe(
        periods=len(test_data),
        freq='W')

    forecast = prophet.predict(future_frame)

    prophet.plot(forecast)

    forecast = forecast[['ds', 'yhat']].yhat \
        .apply(lambda x: int(x))


    return forecast[len(train_data):]

def submission_file_generation(forecast_sj, forecast_iq, sub_name):
    test_index = test[['city', 'year', 'weekofyear']]

    output = pd.concat([forecast_sj, forecast_iq]) \
        .reset_index().drop(['index'], axis=1)

    output.columns = ['total_cases']

    pd.concat([test_index, output], axis=1) \
        .set_index(['city']).to_csv(sub_name + '.csv')

if __name__ == "__main__":	
	train = pd.read_csv('dengue_features_train.csv')
	test = pd.read_csv('dengue_features_test.csv')
	labels = pd.read_csv('dengue_labels_train.csv')

	train['total_cases'] = labels['total_cases']
	_data = train[['city', 'week_start_date', 'total_cases']]

	_data_sj = _data[_data['city'] == 'sj']\
					.drop('city', axis=1)
	_data_iq = _data[_data['city'] == 'iq']\
					.drop('city', axis=1)

	test_sj = test[test['city'] == 'sj']['week_start_date']
	test_iq = test[test['city'] == 'iq']['week_start_date']

	_data_sj.columns, _data_iq.columns = ['ds', 'y'], ['ds', 'y']
	
	_change_point_prior_scale_sj = 0.1
	_change_point_prior_scale_iq = 0.1
	_yearly_seasonality_sj = 10
	_yearly_seasonality_iq = 5

	forecast_sj = additive_model(_data_sj, test_sj, _change_point_prior_scale_sj, _yearly_seasonality_sj)
	forecast_iq = additive_model(_data_iq, test_iq, _change_point_prior_scale_iq, _yearly_seasonality_iq)
	submission_file_generation(forecast_sj, forecast_iq, 'submission_format')
