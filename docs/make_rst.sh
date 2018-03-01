#!/bin/bash
################################################################################
## Bash file for the automatic creation of .RST file                          ##
##                                                                            ##
## This script does not work on modules that not contains class definition    ##
################################################################################

module='ezyrb'

files=`ls ../$module/[a-z]*.py`

for f in $files; do

    filename=$(basename $f .py)

    classname=`grep "class.*(.*):$" $f | awk '{print $2}' | cut -d"(" -f 1`
    [[ -z $classname ]] && echo "WARNING: class not found in file $f" 

    methodnames=`grep -e 'def .*(.*):$' $f | awk '{print $2}' | cut -d "(" -f 1`
    [[ -z $classname ]] && echo "WARNING: methods not found in file $f"

    methodnames=`sed -r 's/\b(__init__|__new__)\b//g' <<< $methodnames`

    output="source/$filename.rst"

    echo -e "$classname"                           >  $output
    echo -e "====================="                >> $output
    echo -e ""                                     >> $output
    echo -e ".. currentmodule:: $module.$filename" >> $output
    echo -e ""                                     >> $output
    echo -e ".. automodule:: $module.$filename"    >> $output
    echo -e ""                                     >> $output
    echo -e ".. autosummary::"                     >> $output
    echo -e "    :toctree: _summaries"             >> $output
    echo -e "    :nosignatures:"                   >> $output
    echo -e ""                                     >> $output
    echo -e "    $classname"                       >> $output
    for methodname in $methodnames; do
        echo -e "    $classname.$methodname"           >> $output
    done
    echo -e ""                                     >> $output
    echo -e ".. autoclass:: $classname"            >> $output
    echo -e "    :members:"                        >> $output
    echo -e "    :private-members:"                >> $output
    echo -e "    :undoc-members:"                  >> $output
    echo -e "    :show-inheritance:"               >> $output
    echo -e "    :noindex:"                        >> $output
done
