################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
DSPLib/source/utility/%.obj: ../DSPLib/source/utility/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: MSP430 Compiler'
	"/Applications/ti/ccs1250/ccs/tools/compiler/ti-cgt-msp430_21.6.1.LTS/bin/cl430" -vmspx --data_model=large --use_hw_mpy=F5 --include_path="/Applications/ti/ccs1250/ccs/ccs_base/msp430/include" --include_path="/Users/lezhang/Documents/Capuchin/capuchin-MCU/DSPLib/include" --include_path="/Users/lezhang/Documents/Capuchin/capuchin-MCU" --include_path="/Applications/ti/ccs1250/ccs/tools/compiler/ti-cgt-msp430_21.6.1.LTS/include" --advice:power="all" --advice:hw_config="all" --define=__MSP430FR5994__ -g --printf_support=minimal --diag_warning=225 --diag_wrap=off --display_error_number --silicon_errata=CPU21 --silicon_errata=CPU22 --silicon_errata=CPU40 --preproc_with_compile --preproc_dependency="DSPLib/source/utility/$(basename $(<F)).d_raw" --obj_directory="DSPLib/source/utility" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


