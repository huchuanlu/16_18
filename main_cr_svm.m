function main_cr_svm

dataBase = 'IP'
winsz    = 3;
%
for num = [5,10,20,30,40,50]
    num
    results = cr_svm(dataBase, num, winsz);
    save([dataBase '_' num2str(winsz) '_' num2str(num) '.mat'], 'results');
end
