echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r AsyncDiff xDiT



echo ""
echo "########## Clone AsyncDiff ##########"
echo ""
bash scripts/clone_asyncdiff_repo.sh
pip install -r AsyncDiff/requirements.txt



echo ""
echo "########## Installing Requirements ##########"
echo ""
pip install -r requirements.txt
