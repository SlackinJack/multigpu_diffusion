echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r AsyncDiff



echo ""
echo "########## Clone AsyncDiff ##########"
echo ""
git clone https://github.com/SlackinJack/AsyncDiff AsyncDiff --depth=1



echo ""
echo "########## Installing Requirements ##########"
echo ""
pip install -r AsyncDiff/requirements.txt
pip install -r requirements.txt
