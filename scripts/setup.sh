echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r AsyncDiff DistriFuser Wan2_1 xDiT



echo ""
echo "########## Clone AsyncDiff ##########"
echo ""
bash scripts/clone_asyncdiff_repo.sh

echo ""
echo "########## Patching AsyncDiff ##########"
echo ""
python3 scripts/PatchAsyncDiff.py



echo ""
echo "########## Clone DistriFuser ##########"
echo ""
bash scripts/clone_distrifuser_repo.sh

echo ""
echo "########## Patching DistriFuser ##########"
echo ""
python3 scripts/PatchDistriFuser.py



echo ""
echo "########## Clone Wan2.1 ##########"
echo ""
bash scripts/clone_wan_repo.sh

echo ""
echo "########## Patching WAN2.1 ##########"
echo ""
python3 scripts/PatchWan.py



echo ""
echo "########## Clone xDiT ##########"
echo ""
bash scripts/clone_xdit_repo.sh

echo ""
echo "########## Patching xDiT ##########"
echo ""
python3 scripts/PatchxDiT.py

