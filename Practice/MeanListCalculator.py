from  statistics import mean


List_of_Off = [6 , 4, 4, 8]

Mean = mean(List_of_Off)
print("mean ",Mean, '\n')

squared_Error= []
# for Point_A, Point_B in zip(List_of_Off, Mean):
#     error = abs(Point_A - Point_B)
#     squared_Error.append(error)
#
# print(squared_Error, '\n')

for point in List_of_Off:
    error = (point - Mean) ** 2
    squared_Error.append(error)
    print("squared error:",squared_Error, '\n')


MSE = mean(squared_Error)
print("mean squared error : ",MSE, '\n')



