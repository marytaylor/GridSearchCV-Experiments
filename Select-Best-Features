# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
  rfc(random_state=1), param_grid={'max_depth': [2, 5, 10]})
best_value = grid_search.fit(
  X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(
  random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)
