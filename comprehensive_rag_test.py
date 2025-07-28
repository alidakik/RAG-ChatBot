"""
Comprehensive Test Suite for RAG-based Project Management System
Tests all documented features across Leads and Jobs for Repair, Tiling, and Excavation work
"""

import subprocess
import sys
import os
from langchain_openai import ChatOpenAI

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Evaluation Task: Determine if the actual response semantically matches the expected response.

CRITICAL EVALUATION CRITERIA:
1. SEMANTIC SIMILARITY is the primary factor - exact wording is NOT required
2. Focus on whether the CORE INFORMATION and MEANING align
3. Accept different phrasing, synonyms, and paraphrasing if the essence is the same
4. Ignore minor differences in formatting, structure, or presentation
5. Numbers can be expressed differently (e.g., "6" vs "six" vs "6 steps")
6. Accept both positive and negative confirmations (e.g., "Yes, X" vs "X is possible")

SPECIFIC MATCHING RULES:
- If expected mentions specific steps/processes, actual should contain those key steps
- If expected lists roles/permissions, actual should identify the same roles (synonyms OK)
- If expected states requirements, actual should mention the same requirements
- If expected gives a yes/no answer, actual should convey the same conclusion
- If expected says "cannot answer", actual should indicate limitation or lack of information

Answer with ONLY 'true' if the actual response semantically matches and contains the key information from the expected response, or 'false' if it contradicts or lacks the essential information.

Answer: """


def query_rag(question: str) -> str:
    """Query the RAG system using the main.py chat command"""
    try:
        result = subprocess.run(
            ["python", "main.py", "chat", "--question", question],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract just the answer part (after the "→" symbol)
        lines = result.stdout.strip().split('\n')
        answer_lines = []
        collecting_answer = False
        
        for line in lines:
            if '→' in line:
                answer_lines.append(line.split('→', 1)[1].strip())
                collecting_answer = True
            elif collecting_answer and line.strip() and not line.startswith('🔹'):
                answer_lines.append(line.strip())
            elif line.startswith('🔹') and collecting_answer:
                break
        
        return ' '.join(answer_lines).strip()
        
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


def query_and_validate(question: str, expected_response: str) -> bool:
    """Query the RAG system and validate the response using OpenAI ChatGPT"""
    response_text = query_rag(question)
    
    if response_text.startswith("Error:"):
        print(f"\n❌ RAG Query Error: {response_text}")
        return False
    
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    try:
        evaluation_results_str = model.invoke(prompt).content
        evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_response}")
        print(f"Actual: {response_text}")
        print(f"Evaluation: {evaluation_results_str_cleaned}")

        if "true" in evaluation_results_str_cleaned:
            print("\033[92m" + f"✅ PASS" + "\033[0m")
            return True
        elif "false" in evaluation_results_str_cleaned:
            print("\033[91m" + f"❌ FAIL" + "\033[0m")
            return False
        else:
            print("\033[93m" + f"⚠️  UNCLEAR: {evaluation_results_str_cleaned}" + "\033[0m")
            return False
            
    except Exception as e:
        print(f"\n💥 Evaluation Error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# REPAIR LEAD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_repair_lead_steps():
    """Test steps to add a repair lead"""
    assert query_and_validate(
        question="How do you add a repair lead?",
        expected_response="Click on leads in sidebar, click Repair, press Add Lead, choose Add Manually or Select from Contacts, choose lead type, optionally add description/phone/location, press Submit",
    )

def test_add_repair_lead_roles():
    """Test which roles can add repair leads"""
    assert query_and_validate(
        question="Which roles can add a repair lead?",
        expected_response="Admin and Project Manager",
    )

def test_repair_lead_required_fields():
    """Test required fields for repair leads"""
    assert query_and_validate(
        question="What are the required fields for creating a repair lead?",
        expected_response="Customer Name and lead type",
    )

def test_repair_lead_optional_fields():
    """Test optional fields for repair leads"""
    assert query_and_validate(
        question="What optional fields can be added to repair leads?",
        expected_response="Description, Phone Number, and Draw/Edit Boundary & Set Location",
    )

def test_repair_lead_equipment_restriction():
    """Test that equipment cannot be added to repair leads"""
    assert query_and_validate(
        question="Can equipment be added to repair leads?",
        expected_response="No, equipment cannot be added to Repair Leads",
    )

def test_repair_lead_designer_restriction():
    """Test that designers cannot be added to repair leads"""
    assert query_and_validate(
        question="Can designers be added to repair leads?",
        expected_response="No, designers cannot be added to Repair Leads",
    )

def test_repair_lead_access_methods():
    """Test how to access repair lead details"""
    assert query_and_validate(
        question="How can you access repair lead details?",
        expected_response="Double clicking on the lead or pressing the 3 points at the end and selecting View Details",
    )

def test_repair_lead_archiving():
    """Test repair lead archiving capability"""
    assert query_and_validate(
        question="Can repair leads be archived?",
        expected_response="Yes, repair leads can be archived",
    )

def test_repair_lead_deletion_policy():
    """Test repair lead deletion policy"""
    assert query_and_validate(
        question="What happens when a repair lead is deleted?",
        expected_response="It is moved to trash page and deleted permanently after 30 days",
    )

def test_repair_lead_811_call():
    """Test 811 call functionality"""
    assert query_and_validate(
        question="Can 811 calls be performed on repair leads?",
        expected_response="Yes, 811 Call can be performed by pressing One Call button",
    )

def test_repair_lead_status_changes():
    """Test repair lead status modification"""
    assert query_and_validate(
        question="Can the status of repair leads be changed?",
        expected_response="Yes, lead status can be changed",
    )

def test_repair_lead_notes_comments():
    """Test adding notes and comments to repair leads"""
    assert query_and_validate(
        question="Can notes and comments be added to repair leads?",
        expected_response="Yes, notes and comments can be added to repair leads",
    )

def test_repair_lead_file_management():
    """Test file upload/delete capability"""
    assert query_and_validate(
        question="Can files be uploaded to repair leads?",
        expected_response="Yes, files can be uploaded to repair leads",
    )

def test_repair_lead_customer_info_editing():
    """Test customer information editing"""
    assert query_and_validate(
        question="Can customer information be edited in repair leads?",
        expected_response="Yes, customer information can be edited in repair leads",
    )

def test_repair_lead_location_editing():
    """Test location and boundary editing"""
    assert query_and_validate(
        question="Can land boundaries and location be edited in repair leads?",
        expected_response="Yes, land boundaries and location can be edited in repair leads",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EXCAVATION LEAD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_excavation_lead_steps():
    """Test steps to add an excavation lead"""
    assert query_and_validate(
        question="How do you add an excavation lead?",
        expected_response="Click on leads in sidebar, click Excavation, press Add Lead, choose Add Manually or Select from Contacts, choose lead type, optionally add description/phone/location, press Submit",
    )

def test_add_excavation_lead_roles():
    """Test which roles can add excavation leads"""
    assert query_and_validate(
        question="Which roles can add excavation leads?",
        expected_response="Admin and Project Manager",
    )

def test_excavation_lead_equipment_designer_restriction():
    """Test that equipment and designers cannot be added to excavation leads"""
    assert query_and_validate(
        question="Can equipment be added to excavation leads?",
        expected_response="No, equipment cannot be added to Excavation Leads",
    )

def test_excavation_lead_status_range():
    """Test excavation lead status options"""
    assert query_and_validate(
        question="What status options are available for excavation leads?",
        expected_response="Status can be changed from new through to completed",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TILING LEAD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_tiling_lead_steps():
    """Test steps to add a tiling lead"""
    assert query_and_validate(
        question="How do you add a tiling lead?",
        expected_response="Click on leads in sidebar, click Tiling, press Add Lead, choose Add Manually or Select from Contacts, choose lead type, optionally add description/phone/location, press Submit",
    )

def test_add_tiling_lead_roles():
    """Test which roles can add tiling leads"""
    assert query_and_validate(
        question="Which roles can add tiling leads?",
        expected_response="Admin and Project Manager",
    )

def test_tiling_lead_equipment_designer_restriction():
    """Test that equipment and designers cannot be added to tiling leads"""
    assert query_and_validate(
        question="Can equipment be added to tiling leads?",
        expected_response="No, equipment cannot be added to Tiling Leads",
    )

def test_tiling_lead_designer_restriction():
    """Test that designers cannot be added to tiling leads"""
    assert query_and_validate(
        question="Can designers be added to tiling leads?",
        expected_response="yes, designers can be added to Tiling Leads",
    )

def test_excavation_lead_designer_restriction():
    """Test that designers cannot be added to excavation leads"""
    assert query_and_validate(
        question="Can designers be added to excavation leads?",
        expected_response="No, designers cannot be added to Excavation Leads",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LEAD CONVERSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_convert_repair_lead_requirements():
    """Test repair lead conversion requirements"""
    assert query_and_validate(
        question="What requirements are needed to convert a repair lead to a job?",
        expected_response="Converting a Repair Lead requires no additional steps or requirements",
    )

def test_convert_repair_lead_steps():
    """Test repair lead conversion steps"""
    assert query_and_validate(
        question="What are the steps to convert a repair lead to a job?",
        expected_response="Click on leads page, click Repair, double click on lead or press 3 points and View Details, press Convert to job button, press convert in popup",
    )

def test_convert_tiling_lead_requirements():
    """Test tiling lead conversion requirements"""
    assert query_and_validate(
        question="What is required when converting a tiling lead to a job?",
        expected_response="Converting a Tiling Lead requires assigning a designer to the job or a designer already assigned to tiling lead",
    )

def test_convert_tiling_lead_steps():
    """Test tiling lead conversion steps"""
    assert query_and_validate(
        question="What are the steps to convert a tiling lead to a job?",
        expected_response="Click on leads page, click Tiling, double click on lead or press 3 points and View Details, press Convert to job button, assign designer and press convert in popup",
    )

def test_convert_excavation_lead_requirements():
    """Test excavation lead conversion requirements"""
    assert query_and_validate(
        question="What is required when converting an excavation lead to a job?",
        expected_response="Converting an Excavation Lead requires adding equipment to the job",
    )

def test_convert_excavation_lead_steps():
    """Test excavation lead conversion steps"""
    assert query_and_validate(
        question="What are the steps to convert an excavation lead to a job?",
        expected_response="Click on leads page, click Excavation, double click on lead or press 3 points and View Details, press Convert to job button, add equipment and press convert in popup",
    )

def test_lead_conversion_data_transfer():
    """Test that data transfers during conversion"""
    assert query_and_validate(
        question="What happens to lead data when converting to a job?",
        expected_response="All files, details, and info from the lead are moved to the job",
    )

def test_job_reversion_restriction():
    """Test that jobs cannot be reverted to leads"""
    assert query_and_validate(
        question="Can jobs be reverted back to leads?",
        expected_response="No, jobs cannot be reverted back to leads",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# JOB CREATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_repair_job_steps():
    """Test steps to add a repair job"""
    assert query_and_validate(
        question="How do you add a repair job?",
        expected_response="Click on Jobs in sidebar, click Repair, press Add Job, choose Add Manually or Select from Contacts, optionally add description/phone/location, press Submit Or by converting lead to job",
    )

def test_add_repair_job_roles():
    """Test which roles can add repair jobs"""
    assert query_and_validate(
        question="Which roles can add repair jobs?",
        expected_response="Admin and Project Manager",
    )

def test_repair_job_required_fields():
    """Test required fields for repair jobs"""
    assert query_and_validate(
        question="What are the required fields for creating a repair job?",
        expected_response="Customer Name is required",
    )

def test_repair_job_equipment_designer_restriction():
    """Test that equipment and designers cannot be added to repair jobs"""
    assert query_and_validate(
        question="Can equipment be added to repair jobs?",
        expected_response="No, equipment cannot be added to Repair Jobs",
    )

def test_repair_job_designer_restriction():
    """Test that designers cannot be added to repair jobs"""
    assert query_and_validate(
        question="Can designers be added to repair jobs?",
        expected_response="No, designers cannot be added to Repair Jobs",
    )

def test_add_excavation_job_steps():
    """Test steps to add an excavation job"""
    assert query_and_validate(
        question="How do you add an excavation job?",
        expected_response="Click on Jobs in sidebar, click Excavation, press Add Job, choose Add Manually or Select from Contacts, press Add Equipment to add equipment, optionally add description/phone/location, press Submit Or by converting lead to job",
    )

def test_excavation_job_equipment_capability():
    """Test that equipment can be added to excavation jobs"""
    assert query_and_validate(
        question="Can equipment be added to excavation jobs?",
        expected_response="Yes, equipment can be added to excavation jobs",
    )

def test_excavation_job_designer_restriction():
    """Test that designers cannot be added to excavation jobs"""
    assert query_and_validate(
        question="Can designers be added to excavation jobs?",
        expected_response="No, designers cannot be added to Excavation Jobs",
    )

def test_excavation_job_hours_setting():
    """Test hours setting for excavation jobs"""
    assert query_and_validate(
        question="Can hours be set for working machines in excavation jobs?",
        expected_response="Yes, hours can be set for working machines",
    )

def test_add_tiling_job_steps():
    """Test steps to add a tiling job"""
    assert query_and_validate(
        question="How do you add a tiling job?",
        expected_response="Click on Jobs in sidebar, click Tiling, press Add Job, choose Add Manually or Select from Contacts, press Add Machines to add machines, optionally select Project Crew and Designer and add description/phone/acre/location, press Submit Or by converting lead to job",
    )

def test_tiling_job_machines_capability():
    """Test that machines can be added to tiling jobs"""
    assert query_and_validate(
        question="Can machines be added to tiling jobs?",
        expected_response="Yes, machines can be added to tiling jobs",
    )

def test_tiling_job_designer_capability():
    """Test that designers can be added to tiling jobs"""
    assert query_and_validate(
        question="Can designers be added to tiling jobs?",
        expected_response="Yes, designers can be added to tiling jobs",
    )

def test_tiling_job_project_crew():
    """Test project crew assignment in tiling jobs"""
    assert query_and_validate(
        question="Can project crew be assigned to tiling jobs?",
        expected_response="Yes, Project Crew can be selected for tiling jobs",
    )

def test_tiling_job_vehicle_restriction():
    """Test vehicle restriction for tiling jobs"""
    assert query_and_validate(
        question="Can vehicles be added to tiling jobs?",
        expected_response="No, vehicles cannot be added to Tiling Jobs",
    )

def test_tiling_job_installed_footage():
    """Test installed footage feature"""
    assert query_and_validate(
        question="Can installed footage be tracked in tiling jobs?",
        expected_response="Yes, installed footage can be added and downloaded as an Excel Sheet",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INVOICE AND BUSINESS FEATURES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_repair_job_invoicing():
    """Test invoice creation for repair jobs"""
    assert query_and_validate(
        question="Can invoices be created for repair jobs?",
        expected_response="Yes, invoices can be created for repair jobs in Book Keeping page by pressing Invoice button",
    )

def test_excavation_job_invoicing():
    """Test invoice creation for excavation jobs"""
    assert query_and_validate(
        question="Can invoices be created for excavation jobs?",
        expected_response="Yes, invoices can be created for excavation jobs in Book Keeping page by pressing Invoice button",
    )

def test_tiling_job_invoicing():
    """Test invoice creation for tiling jobs"""
    assert query_and_validate(
        question="Can invoices be created for tiling jobs?",
        expected_response="Yes, invoices can be created for tiling jobs in Book Keeping page by pressing Invoice button",
    )

def test_tiling_job_pipe_ordering():
    """Test pipe ordering for tiling jobs"""
    assert query_and_validate(
        question="Can pipes be ordered for tiling jobs?",
        expected_response="Yes, pipes can be ordered for Tiling Jobs in Order Pipes page by pressing Order Pipes button",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INVOICE CREATION TESTS (DOC10)
# ═══════════════════════════════════════════════════════════════════════════════

def test_invoice_creation_steps():
    """Test steps to create an invoice"""
    assert query_and_validate(
        question="How do you create an invoice?",
        expected_response="Click on Jobs in sidebar, click designated section (Repair/Excavation/Tiling), choose the job that needs invoice, press Invoice and yes to confirmation popup, redirected to Book Keeping page, fill client info, create invoice items by filling Activity/Description/Unit Price, press Add Order, press Update Bill To to send to client",
    )

def test_invoice_creation_roles():
    """Test which roles can create invoices"""
    assert query_and_validate(
        question="Which roles can create invoices?",
        expected_response="Admin and Bookkeeper",
    )

def test_invoice_creation_location():
    """Test where invoice creation happens"""
    assert query_and_validate(
        question="Where can invoices be created?",
        expected_response="Book Keeping page",
    )

def test_invoice_job_types():
    """Test which job types can have invoices"""
    assert query_and_validate(
        question="For which job types can invoices be created?",
        expected_response="Invoices can only be created for Jobs (Repair/Excavation/Tiling)",
    )

def test_invoice_client_info_fields():
    """Test required client information for invoices"""
    assert query_and_validate(
        question="What client information is needed when creating an invoice?",
        expected_response="Client Name/Company, Client Address, Client Contact, Description, Due Date",
    )

def test_invoice_item_fields():
    """Test invoice item creation fields"""
    assert query_and_validate(
        question="What fields are needed to create invoice items?",
        expected_response="Activity, Description, Unit Price, and Unit Price",
    )

def test_invoice_admin_check_requirement():
    """Test admin check requirement for updating bill"""
    assert query_and_validate(
        question="What is required before pressing the Update Bill button?",
        expected_response="The Checked by Admin button should be pressed by an admin",
    )

def test_invoice_admin_check_restriction():
    """Test who can press the Checked by Admin button"""
    assert query_and_validate(
        question="Who can press the Checked by Admin button?",
        expected_response="Only admins can press the Checked by Admin button",
    )

def test_invoice_sent_to_client_workflow():
    """Test the sent to client workflow"""
    assert query_and_validate(
        question="What happens after an invoice is sent to the client?",
        expected_response="Once sent, the Admin or Book Keeper must press the Sent to Client button",
    )

def test_invoice_payment_workflow():
    """Test the payment workflow"""
    assert query_and_validate(
        question="What happens when an invoice is paid?",
        expected_response="Once the invoice is paid, the Admin or Book Keeper must press the Paid button",
    )

def test_invoice_multiple_per_job():
    """Test multiple invoices per job capability"""
    assert query_and_validate(
        question="Can multiple invoices be created for the same job?",
        expected_response="Yes, multiple invoices can be created for the same job",
    )

def test_invoice_multiple_orders():
    """Test multiple orders per invoice capability"""
    assert query_and_validate(
        question="Can an invoice contain multiple orders?",
        expected_response="Yes, an invoice can contain multiple orders",
    )

def test_invoice_deletion():
    """Test invoice deletion capability"""
    assert query_and_validate(
        question="Can invoices be deleted?",
        expected_response="Yes, an invoice can be deleted",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER PIPES TESTS (DOC11)
# ═══════════════════════════════════════════════════════════════════════════════

def test_order_pipes_steps():
    """Test steps to create a pipe order"""
    assert query_and_validate(
        question="How do you create a pipe order?",
        expected_response="Click on Jobs in sidebar, click Tiling, choose designated Tiling job (double click or press three points and View details), press Order Pipes button, press yes in confirmation popup, redirected to order pipes page, add delivery location in Order Details section, create invoice items by adding Main Category and quantity, press Add Order, share to vendor by pressing Share to Vendor button",
    )

def test_order_pipes_roles():
    """Test which roles can create pipe orders"""
    assert query_and_validate(
        question="Which roles can create pipe orders?",
        expected_response="Admin and Project Manager",
    )

def test_order_pipes_location():
    """Test where pipe ordering happens"""
    assert query_and_validate(
        question="Where can pipe orders be created?",
        expected_response="Order Pipes page",
    )

def test_order_pipes_job_restriction():
    """Test which jobs can have pipe orders"""
    assert query_and_validate(
        question="Which jobs can have pipe orders?",
        expected_response="Order pipes option is only present in tiling jobs",
    )

def test_order_pipes_repair_excavation_restriction():
    """Test that repair and excavation jobs cannot have pipe orders"""
    assert query_and_validate(
        question="Can repair or excavation jobs have pipe orders?",
        expected_response="Order pipes option is not present in neither repair nor excavation jobs",
    )

def test_order_pipes_leads_restriction():
    """Test that leads cannot have pipe orders"""
    assert query_and_validate(
        question="Can leads have pipe orders?",
        expected_response="Order pipes option is not present in all leads",
    )

def test_order_pipes_vendor_restriction():
    """Test vendor sharing restrictions"""
    assert query_and_validate(
        question="Can orders be shared to any vendor?",
        expected_response="The order can only be shared to the vendors enlisted on the system",
    )

def test_order_pipes_stages():
    """Test order processing stages"""
    assert query_and_validate(
        question="What stages does a shared order go through?",
        expected_response="Once the order is shared, it passes through five stages: Pending, Shared, Received, Accepted, and Delivered",
    )

def test_order_pipes_received_stage():
    """Test the received stage"""
    assert query_and_validate(
        question="When does an order become received?",
        expected_response="Once the vendor gets the order and checks it out, it becomes received",
    )

def test_order_pipes_accepted_stage():
    """Test the accepted stage"""
    assert query_and_validate(
        question="When does an order become accepted?",
        expected_response="Once the vendor accepts the order and starts the packing of the items, it becomes accepted",
    )

def test_order_pipes_delivered_stage():
    """Test the delivered stage"""
    assert query_and_validate(
        question="When does an order become delivered?",
        expected_response="Once the order is delivered to the assigned location, it becomes delivered",
    )

def test_order_pipes_delivery_location():
    """Test delivery location requirement"""
    assert query_and_validate(
        question="What needs to be specified in the Order Details section?",
        expected_response="Add the delivery location in the Order Details section",
    )

def test_order_pipes_item_fields():
    """Test order item creation fields"""
    assert query_and_validate(
        question="What information is needed to create order items?",
        expected_response="Main Category and the quantity",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EQUIPMENT MACHINES TESTS (DOC12)
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_machine_steps():
    """Test steps to add a machine"""
    assert query_and_validate(
        question="How do you add a machine?",
        expected_response="Click on Equipment in sidebar, click Add Equipment, choose Machines, fill required details (Machine Name, Assign a User, Current Hours, Hourly rate, tracker status), optionally fill additional information (Serial number, Machine image) and add filters, press Add Machine",
    )

def test_add_machine_roles():
    """Test which roles can add machines"""
    assert query_and_validate(
        question="Which roles can add machines?",
        expected_response="Admin and Project Manager",
    )

def test_machine_required_fields():
    """Test required fields for adding machines"""
    assert query_and_validate(
        question="What are the required fields for adding a machine?",
        expected_response="Machine Name, Assign a User, Current Hours, Hourly rate, and tracker status",
    )

def test_machine_optional_fields():
    """Test optional fields for machines"""
    assert query_and_validate(
        question="What optional information can be added to machines?",
        expected_response="Serial number, Machine image",
    )

def test_machine_filter_options():
    """Test machine filter options"""
    assert query_and_validate(
        question="What filters can be added to machines?",
        expected_response="Fuel Filter, Air Filter, Oil Filter, Hydraulic Filter, Secondary Hydraulic Filter, Hydraulic Water Separator, Hydraulic Return Filter, Final Oil Drive",
    )

def test_machine_job_restrictions():
    """Test which jobs can have machines"""
    assert query_and_validate(
        question="Which jobs can have machines added?",
        expected_response="Machines can only be added to Excavation and Tiling jobs",
    )

def test_machine_repair_restriction():
    """Test that machines cannot be added to repair jobs or leads"""
    assert query_and_validate(
        question="Can machines be added to repair jobs or leads?",
        expected_response="Machines can neither be added to Repair jobs nor leads",
    )

def test_machine_filtering_view():
    """Test filtering machines in equipment page"""
    assert query_and_validate(
        question="How do you view only machines in the equipment page?",
        expected_response="Press on Type Filter and choose Machines. To remove the filter, press Type Filter and press Clear All",
    )

def test_machine_maintenance_status():
    """Test machine maintenance status change"""
    assert query_and_validate(
        question="What happens to machine status during maintenance?",
        expected_response="When the machine is in maintenance, the status will be changed from Available to Unavailable",
    )

def test_machine_access_methods():
    """Test how to access machine details"""
    assert query_and_validate(
        question="How do you access a machine in the equipment page?",
        expected_response="Double click on the machine or press the three points on the left, View details",
    )

def test_machine_editing_capability():
    """Test machine editing capabilities"""
    assert query_and_validate(
        question="Can machine information be edited?",
        expected_response="Machine info can be edited from inside the machine. You can add or remove filters from inside the machine. Operation hours can be edited from inside the machine",
    )

def test_machine_deletion():
    """Test machine deletion"""
    assert query_and_validate(
        question="How can machines be deleted?",
        expected_response="Machines can be deleted by pressing the three points on the left then delete",
    )

def test_machine_filter_management():
    """Test adding and managing filters inside machines"""
    assert query_and_validate(
        question="How do you add filters inside a machine?",
        expected_response="Press on Add Filters, choose the desired filter, and set the last changed and threshold amounts. Amounts set can then be edited in the machine. Filters can also be deleted by pressing the trash bin icon",
    )

def test_machine_hours_in_jobs():
    """Test setting machine hours in jobs"""
    assert query_and_validate(
        question="Can machine hours be set in jobs?",
        expected_response="Machine hours can also be set in the tiling/excavation job",
    )

def test_machine_maintenance_board():
    """Test adding machines to maintenance board"""
    assert query_and_validate(
        question="How can machines be added to maintenance?",
        expected_response="Machines can also be added to maintenance by pressing Add to Maintenance Board",
    )

def test_machine_automatic_maintenance():
    """Test automatic maintenance trigger"""
    assert query_and_validate(
        question="When are machines automatically moved to maintenance?",
        expected_response="When the difference between current hours of a machine and the last changed hours exceed the threshold of a filter, it is automatically moved to the maintenance page",
    )

def test_machine_maintenance_log():
    """Test machine maintenance history"""
    assert query_and_validate(
        question="How can you view machine maintenance history?",
        expected_response="The maintenance history of a machine can be viewed by pressing Print/Download Maintenance Log in the machine",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EQUIPMENT VEHICLES TESTS (DOC13)
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_vehicle_steps():
    """Test steps to add a vehicle"""
    assert query_and_validate(
        question="How do you add a vehicle?",
        expected_response="Click on Equipment in sidebar, click Add Equipment, choose Vehicles, fill required details (Vehicle Name, Assign a User, Current Miles, tracker status), optionally fill additional information (License Plate, VIN Number, Vehicle Image, Registration Image, Insurance Image) and add filters, press Add Vehicle",
    )

def test_add_vehicle_roles():
    """Test which roles can add vehicles"""
    assert query_and_validate(
        question="Which roles can add vehicles?",
        expected_response="Admin and Project Manager",
    )

def test_vehicle_required_fields():
    """Test required fields for adding vehicles"""
    assert query_and_validate(
        question="What are the required fields for adding a vehicle?",
        expected_response="Vehicle Name, Assign a User, Current Miles, and tracker status",
    )

def test_vehicle_optional_fields():
    """Test optional fields for vehicles"""
    assert query_and_validate(
        question="What optional information can be added to vehicles?",
        expected_response="License Plate, VIN Number, Vehicle Image, Registration Image, Insurance Image",
    )

def test_vehicle_filter_options():
    """Test vehicle filter options"""
    assert query_and_validate(
        question="What filters can be added to vehicles?",
        expected_response="Fuel Filter, Air Filter, Oil Filter, Hydraulic Filter",
    )

def test_vehicle_job_restrictions():
    """Test which jobs can have vehicles"""
    assert query_and_validate(
        question="Which jobs can have vehicles added?",
        expected_response="Vehicles can only be added to Excavation",
    )

def test_vehicle_repair_tiling_restriction():
    """Test that vehicles cannot be added to repair or tiling jobs"""
    assert query_and_validate(
        question="Can vehicles be added to repair jobs, tiling jobs, or leads?",
        expected_response="Vehicles cannot be added to Repair jobs, Tiling jobs, or leads",
    )

def test_vehicle_filtering_view():
    """Test filtering vehicles in equipment page"""
    assert query_and_validate(
        question="How do you view only vehicles in the equipment page?",
        expected_response="Press on Type Filter and choose Vehicles. To remove the filter, press Type Filter and press Clear All",
    )

def test_vehicle_maintenance_status():
    """Test vehicle maintenance status change"""
    assert query_and_validate(
        question="What happens to vehicle status during maintenance?",
        expected_response="When the Vehicle is in maintenance, the status will be changed from Available to Unavailable",
    )

def test_vehicle_access_methods():
    """Test how to access vehicle details"""
    assert query_and_validate(
        question="How do you access a vehicle in the equipment page?",
        expected_response="Double click on the Vehicle or press the three points on the left, View details",
    )

def test_vehicle_editing_capability():
    """Test vehicle editing capabilities"""
    assert query_and_validate(
        question="Can vehicle information be edited?",
        expected_response="Vehicle info can be edited from inside the Vehicle. You can add or remove filters from inside the Vehicle. Mileage can be edited from inside the Vehicle",
    )

def test_vehicle_deletion():
    """Test vehicle deletion"""
    assert query_and_validate(
        question="How can vehicles be deleted?",
        expected_response="Vehicles can be deleted by pressing the three points on the left then delete",
    )

def test_vehicle_filter_management():
    """Test adding and managing filters inside vehicles"""
    assert query_and_validate(
        question="How do you add filters inside a vehicle?",
        expected_response="Press on Add Filters, choose the desired filter, and set the last changed and threshold amounts. Amounts set can then be edited in the Vehicle. Filters can also be deleted by pressing the trash bin icon",
    )

def test_vehicle_mileage_in_jobs():
    """Test setting vehicle mileage in jobs"""
    assert query_and_validate(
        question="Can vehicle mileage be set in jobs?",
        expected_response="Mileage can also be set in the excavation job",
    )

def test_vehicle_automatic_maintenance():
    """Test automatic maintenance trigger for vehicles"""
    assert query_and_validate(
        question="When are vehicles automatically moved to maintenance?",
        expected_response="When the difference between current miles of a Vehicle and the last changed miles exceed the threshold of a filter, it is automatically moved to the maintenance page",
    )

def test_vehicle_maintenance_board():
    """Test adding vehicles to maintenance board"""
    assert query_and_validate(
        question="How can vehicles be added to maintenance?",
        expected_response="Vehicle can also be added to maintenance by pressing Add to Maintenance Board",
    )

def test_vehicle_maintenance_log():
    """Test vehicle maintenance history"""
    assert query_and_validate(
        question="How can you view vehicle maintenance history?",
        expected_response="The maintenance history of a vehicle can be viewed by pressing Print/Download Maintenance Log in the vehicle",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EQUIPMENT TRAILERS TESTS (DOC14)
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_trailer_steps():
    """Test steps to add a trailer"""
    assert query_and_validate(
        question="How do you add a trailer?",
        expected_response="Click on Equipment in sidebar, click Add Equipment, choose Trailers, fill required details (Trailer Name, Assign a User, tracker status), optionally fill additional information (License Plate, Serial Number, Trailer Image), press Add Trailer",
    )

def test_add_trailer_roles():
    """Test which roles can add trailers"""
    assert query_and_validate(
        question="Which roles can add trailers?",
        expected_response="Admin and Project Manager",
    )

def test_trailer_required_fields():
    """Test required fields for adding trailers"""
    assert query_and_validate(
        question="What are the required fields for adding a trailer?",
        expected_response="Trailer Name, Assign a User, and tracker status",
    )

def test_trailer_optional_fields():
    """Test optional fields for trailers"""
    assert query_and_validate(
        question="What optional information can be added to trailers?",
        expected_response="License Plate, Serial Number, and Trailer Image",
    )

def test_trailer_job_restrictions():
    """Test which jobs can have trailers"""
    assert query_and_validate(
        question="Which jobs can have trailers added?",
        expected_response="Trailers can only be added to Excavation",
    )

def test_trailer_repair_tiling_restriction():
    """Test that trailers cannot be added to repair or tiling jobs"""
    assert query_and_validate(
        question="Can trailers be added to repair jobs, tiling jobs, or leads?",
        expected_response="Trailers cannot be added to Repair jobs, Tiling jobs, or leads",
    )

def test_trailer_filtering_view():
    """Test filtering trailers in equipment page"""
    assert query_and_validate(
        question="How do you view only trailers in the equipment page?",
        expected_response="Press on Type Filter and choose Trailers. To remove the filter, press Type Filter and press Clear All",
    )

def test_trailer_maintenance_status():
    """Test trailer maintenance status change"""
    assert query_and_validate(
        question="What happens to trailer status during maintenance?",
        expected_response="When the Trailer is in maintenance, the status will be changed from Available to Unavailable",
    )

def test_trailer_access_methods():
    """Test how to access trailer details"""
    assert query_and_validate(
        question="How do you access a trailer in the equipment page?",
        expected_response="Double click on the Trailer or press the three points on the left, View details",
    )

def test_trailer_editing_capability():
    """Test trailer editing capabilities"""
    assert query_and_validate(
        question="Can trailer information be edited?",
        expected_response="Trailer info can be edited from inside the Trailer",
    )

def test_trailer_deletion():
    """Test trailer deletion"""
    assert query_and_validate(
        question="How can trailers be deleted?",
        expected_response="Trailers can be deleted by pressing the three points on the left then delete",
    )

def test_trailer_maintenance_board():
    """Test adding trailers to maintenance board"""
    assert query_and_validate(
        question="How can trailers be added to maintenance?",
        expected_response="Trailer can also be added to maintenance by pressing Add to Maintenance Board",
    )

def test_trailer_maintenance_log():
    """Test trailer maintenance history"""
    assert query_and_validate(
        question="How can you view trailer maintenance history?",
        expected_response="The maintenance history of a Trailer can be viewed by pressing Print/Download Maintenance Log in the Trailer",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE TESTS (DOC15)
# ═══════════════════════════════════════════════════════════════════════════════

def test_maintenance_steps():
    """Test steps to perform maintenance"""
    assert query_and_validate(
        question="How do you perform maintenance on equipment?",
        expected_response="Click on Maintenance in the sidebar, choose the desired equipment (Machine/Vehicle/Trailer), access the equipment by double clicking on it or by pressing the three points on the left and pressing View Details, press Contact Maintenance after contacting the people assigned to do the required maintenance, press Mark as completed for each issue after it is serviced, press Mark Maintenance as Complete after all the issues are Fixed",
    )

def test_maintenance_roles():
    """Test which roles can perform maintenance"""
    assert query_and_validate(
        question="Which roles can perform maintenance?",
        expected_response="Admin, Project Manager, and Project Crew",
    )

def test_maintenance_location():
    """Test where maintenance is performed"""
    assert query_and_validate(
        question="Where is maintenance performed?",
        expected_response="Maintenance page",
    )

def test_maintenance_equipment_types():
    """Test which equipment types can be maintained"""
    assert query_and_validate(
        question="What types of equipment can be maintained?",
        expected_response="Machine, Vehicle, and Trailer",
    )

def test_maintenance_access_methods():
    """Test how to access equipment for maintenance"""
    assert query_and_validate(
        question="How do you access equipment for maintenance?",
        expected_response="Double clicking on it or by pressing the three points on the left and pressing View Details",
    )

def test_maintenance_contact_requirement():
    """Test contact maintenance requirement"""
    assert query_and_validate(
        question="What is required before marking maintenance issues as complete?",
        expected_response="Contact Maintenance must be pressed by the assigned member before marking issues as complete",
    )

def test_maintenance_admin_privileges():
    """Test admin privileges in maintenance"""
    assert query_and_validate(
        question="Can admins mark maintenance issues as complete without contacting maintenance?",
        expected_response="Yes, only the Admin can press the mark as completed button for issues without pressing Contact Maintenance",
    )

def test_maintenance_assigned_member_change():
    """Test changing assigned member for maintenance"""
    assert query_and_validate(
        question="Can the assigned member for maintenance be changed?",
        expected_response="Yes, assigned member for Maintenance can be changed from inside the Maintenance page, Equipment requiring maintenance",
    )

def test_maintenance_notes_comments():
    """Test adding notes and comments in maintenance"""
    assert query_and_validate(
        question="Can notes and comments be added for equipment in maintenance?",
        expected_response="Yes, notes and comments can be added for Equipment in Maintenance",
    )

def test_maintenance_equipment_deletion():
    """Test deleting equipment from maintenance"""
    assert query_and_validate(
        question="Can equipment be deleted from maintenance?",
        expected_response="Yes, equipment in Maintenance can be deleted by pressing the three points on the left of the equipment requiring maintenance in maintenance page, and pressing Delete",
    )

def test_maintenance_completion_status_change():
    """Test status change after maintenance completion"""
    assert query_and_validate(
        question="What happens to equipment status after maintenance is completed?",
        expected_response="The Equipment will be removed from the maintenance page, and the Status of the Equipment will be changed from Unavailable to Available",
    )

def test_maintenance_manual_addition():
    """Test manually adding equipment to maintenance"""
    assert query_and_validate(
        question="How can equipment be manually added to maintenance?",
        expected_response="Equipment can be added to maintenance manually by pressing Add to Maintenance Board",
    )

def test_maintenance_automatic_machine_addition():
    """Test automatic addition of machines to maintenance"""
    assert query_and_validate(
        question="When are machines automatically added to maintenance?",
        expected_response="Machines can be added to maintenance automatically when the difference between current hours and last changed exceeds the threshold of the filter",
    )

def test_maintenance_automatic_vehicle_addition():
    """Test automatic addition of vehicles to maintenance"""
    assert query_and_validate(
        question="When are vehicles automatically added to maintenance?",
        expected_response="Vehicles can be added to maintenance automatically when the difference between current miles and last changed exceeds the threshold of the filter",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TEAMS TESTS (DOC16)
# ═══════════════════════════════════════════════════════════════════════════════

def test_teams_invitation_steps():
    """Test steps to invite team members"""
    assert query_and_validate(
        question="How do you invite team members?",
        expected_response="Click on Organization profile, click on Teams, press Invite Member to invite new members to the organization, fill the email address of the invited member and select their designated role, press Send Invitation to send an invitation to the member via email",
    )

def test_teams_roles():
    """Test which roles can access teams"""
    assert query_and_validate(
        question="Which roles can access the teams page?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_teams_location():
    """Test where teams functionality is located"""
    assert query_and_validate(
        question="Where is the teams functionality located?",
        expected_response="Teams page",
    )

def test_teams_member_joining():
    """Test how members join the organization"""
    assert query_and_validate(
        question="How do invited members join the organization?",
        expected_response="The member can join the organization through the link sent to them via email",
    )

def test_teams_search_capability():
    """Test member search functionality"""
    assert query_and_validate(
        question="Can team members be searched?",
        expected_response="Yes, members can be searched through the search bar",
    )

def test_teams_role_filtering():
    """Test role filtering functionality"""
    assert query_and_validate(
        question="Can team members be filtered by role?",
        expected_response="Yes, roles can be filtered using the filter All Roles and choosing the desired role",
    )

def test_teams_member_information_viewing():
    """Test what member information can be viewed"""
    assert query_and_validate(
        question="What member information can be viewed?",
        expected_response="Any member can view the member's Name, Role, Phone, and email",
    )

def test_teams_role_change_permissions():
    """Test who can change member roles"""
    assert query_and_validate(
        question="Who can change the role of a member?",
        expected_response="Only the admin can change the role of a member",
    )

def test_teams_member_deletion_permissions():
    """Test who can delete users from organization"""
    assert query_and_validate(
        question="Who can delete a user from the organization?",
        expected_response="Only the admin can delete a user from the organization",
    )

def test_teams_owner_role_restriction():
    """Test owner role change restrictions"""
    assert query_and_validate(
        question="Can admins change the owner's role?",
        expected_response="No, admins cannot change the Owner's role",
    )

def test_teams_invitation_permissions():
    """Test who can make invitations"""
    assert query_and_validate(
        question="Who can make invitations to join the organization?",
        expected_response="Only the admin and owner can make invitation, remaining roles can just see the page",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZATION INFO TESTS (DOC17)
# ═══════════════════════════════════════════════════════════════════════════════

def test_organization_info_access_steps():
    """Test steps to access organization info"""
    assert query_and_validate(
        question="How do you access organization info?",
        expected_response="Click on Organization profile, click on Organization Info",
    )

def test_organization_info_roles():
    """Test which roles can access organization info"""
    assert query_and_validate(
        question="Which roles can access organization info?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_organization_info_location():
    """Test where organization info is located"""
    assert query_and_validate(
        question="Where is organization info located?",
        expected_response="Organization Info page",
    )

def test_organization_info_editing_steps():
    """Test steps to edit organization info"""
    assert query_and_validate(
        question="How do you edit organization info?",
        expected_response="Press Edit to Edit organization details like Logo, Organization Name, Phone Number, and Address, press edit on the left to edit the organization Location",
    )

def test_organization_info_editable_fields():
    """Test what organization details can be edited"""
    assert query_and_validate(
        question="What organization details can be edited?",
        expected_response="Logo, Organization Name, Phone Number, and Address",
    )

def test_organization_info_location_editing():
    """Test organization location editing"""
    assert query_and_validate(
        question="How do you edit organization location?",
        expected_response="Press edit on the left to edit the organization Location",
    )

def test_organization_info_editing_permissions():
    """Test who can edit organization info"""
    assert query_and_validate(
        question="Who can edit organization details and location?",
        expected_response="Only Admins can Edit the organization details and Location",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CANCELLED JOBS TESTS (DOC18)
# ═══════════════════════════════════════════════════════════════════════════════

def test_cancel_job_steps():
    """Test steps to cancel a job"""
    assert query_and_validate(
        question="How do you cancel a job?",
        expected_response="Click on Jobs, choose the designated section (Repair/Excavation/Tiling), enter the job that requires cancelling by double clicking in the job or pressing the 3 points on the left View Details, press on the three points beside the status then choose Canceled, the job is going to be moved to the completed and canceled Jobs",
    )

def test_cancel_job_permissions():
    """Test who can cancel jobs"""
    assert query_and_validate(
        question="Who can cancel a job?",
        expected_response="Only Admins can Cancel a Job",
    )

def test_cancelled_jobs_location():
    """Test where cancelled jobs are located"""
    assert query_and_validate(
        question="Where are cancelled jobs moved to?",
        expected_response="The job is going to be moved to the completed and canceled Jobs",
    )

def test_cancelled_jobs_filtering():
    """Test filtering cancelled jobs"""
    assert query_and_validate(
        question="How can cancelled jobs be filtered?",
        expected_response="Jobs in completed and canceled page can be filtered by pressing Status Filter and choosing Cancelled",
    )

def test_cancelled_jobs_access():
    """Test how to access cancelled jobs"""
    assert query_and_validate(
        question="How can cancelled jobs be accessed?",
        expected_response="Cancelled jobs can be accessed by double clicking on them, or by pressing the three points on the left, then View Details",
    )

def test_cancelled_jobs_content():
    """Test cancelled jobs content and restrictions"""
    assert query_and_validate(
        question="What is different about cancelled jobs compared to regular jobs?",
        expected_response="Cancelled jobs content is the same as a regular job, except status cannot be changed",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# DELETING ACCOUNT TESTS (DOC19)
# ═══════════════════════════════════════════════════════════════════════════════

def test_delete_account_steps():
    """Test steps to delete an account"""
    assert query_and_validate(
        question="How do you delete an account?",
        expected_response="Click on the upper right corner logo or upper left corner logo and access any option for the second sidebar to be apparent, click on Delete account in User Settings section, tic the I understand that this action is permanent and cannot be undone option and type the DELETE word in the box then press Delete Account, the account will be permanently deleted",
    )

def test_delete_account_permissions():
    """Test who can delete an account"""
    assert query_and_validate(
        question="Who can delete an account?",
        expected_response="Only the user can terminate their designated account",
    )

def test_delete_account_location():
    """Test where account deletion is located"""
    assert query_and_validate(
        question="Where is account deletion located?",
        expected_response="Delete account page",
    )

def test_delete_account_confirmation_requirements():
    """Test account deletion confirmation requirements"""
    assert query_and_validate(
        question="What is required to confirm account deletion?",
        expected_response="Tic the I understand that this action is permanent and cannot be undone option and type the DELETE word in the box",
    )

def test_delete_account_consequences_profile():
    """Test consequences of deleting account on profile"""
    assert query_and_validate(
        question="What happens to your profile when you delete your account?",
        expected_response="Your profile and personal information will be permanently deleted",
    )

def test_delete_account_consequences_data():
    """Test consequences of deleting account on data"""
    assert query_and_validate(
        question="What happens to your data when you delete your account as an owner?",
        expected_response="All your data including projects, files, and settings will be removed if you are the owner of an organization",
    )

def test_delete_account_consequences_access():
    """Test consequences of deleting account on access"""
    assert query_and_validate(
        question="What happens to your access when you delete your account?",
        expected_response="You will lose access to all services linked to this account",
    )

def test_delete_account_username_availability():
    """Test username availability after account deletion"""
    assert query_and_validate(
        question="What happens to your username after deleting your account?",
        expected_response="Your username will become available for others to claim",
    )

def test_delete_account_irreversibility():
    """Test account deletion irreversibility"""
    assert query_and_validate(
        question="Can account deletion be undone?",
        expected_response="This action is irreversible and cannot be undone",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TRASH PAGE TESTS (DOC20)
# ═══════════════════════════════════════════════════════════════════════════════

def test_trash_page_access_steps():
    """Test steps to access trash page"""
    assert query_and_validate(
        question="How do you access the trash page?",
        expected_response="Click on the upper right corner logo, click on trash, now you can view deleted jobs and equipment",
    )

def test_trash_page_roles():
    """Test which roles can access trash page"""
    assert query_and_validate(
        question="Which roles can access the trash page?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_trash_page_content():
    """Test what content is in trash page"""
    assert query_and_validate(
        question="What can be viewed in the trash page?",
        expected_response="Deleted jobs and equipment",
    )

def test_trash_page_automatic_movement():
    """Test automatic movement to trash"""
    assert query_and_validate(
        question="What happens to deleted jobs and equipment?",
        expected_response="Deleted jobs and equipment are always moved to trash page",
    )

def test_trash_page_permanent_deletion_time():
    """Test permanent deletion timeframe"""
    assert query_and_validate(
        question="How long do items stay in trash before permanent deletion?",
        expected_response="After 30 days, the jobs/equipment will be permanently deleted",
    )

def test_trash_page_item_access():
    """Test how to access items in trash"""
    assert query_and_validate(
        question="How can you access items in the trash page?",
        expected_response="You can access the items found in trash page by double clicking on them, or by pressing the three points on the left, then View Details",
    )

def test_trash_page_item_restoration():
    """Test item restoration from trash"""
    assert query_and_validate(
        question="How can items be restored from trash?",
        expected_response="Items can be restored by pressing restore inside it",
    )

def test_trash_page_permanent_deletion():
    """Test permanent deletion from trash"""
    assert query_and_validate(
        question="How can items be permanently deleted from trash?",
        expected_response="Items can be permanently deleted by pressing delete inside it",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGING SYSTEM TESTS (DOC21)
# ═══════════════════════════════════════════════════════════════════════════════

def test_messaging_system_access():
    """Test how to access messaging system"""
    assert query_and_validate(
        question="How do you access the messaging system?",
        expected_response="Click on the letter icon in the upper bar, now you can access the messages",
    )

def test_messaging_system_roles():
    """Test which roles can use messaging system"""
    assert query_and_validate(
        question="Which roles can use the messaging system?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_messaging_system_location():
    """Test where messaging system is located"""
    assert query_and_validate(
        question="Where is the messaging system located?",
        expected_response="Messages",
    )

def test_messaging_group_creation_steps():
    """Test steps to create a group message"""
    assert query_and_validate(
        question="How do you create a group message?",
        expected_response="Click on Add Group, fill the group name and tic the desired members, click on Create Group, the group is now created in the Groups section",
    )

def test_messaging_unread_indicator():
    """Test unread messages indicator"""
    assert query_and_validate(
        question="How are unread messages indicated?",
        expected_response="The number of messages not read is shown as a small number on the letter icon",
    )

def test_messaging_multiple_groups():
    """Test multiple group creation capability"""
    assert query_and_validate(
        question="Can you create multiple group messages?",
        expected_response="Yes, you can create multiple group messages",
    )

def test_messaging_member_multiple_groups():
    """Test member participation in multiple groups"""
    assert query_and_validate(
        question="Can a member be in multiple groups?",
        expected_response="Yes, a member can be in multiple groups at the same time",
    )

def test_messaging_message_types():
    """Test types of messages that can be sent"""
    assert query_and_validate(
        question="What types of messages can be sent?",
        expected_response="A message can be letters, audio recording, image, or files",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TESTS (DOC22)
# ═══════════════════════════════════════════════════════════════════════════════

def test_dashboard_access():
    """Test how to access dashboard"""
    assert query_and_validate(
        question="How do you access the dashboard?",
        expected_response="Click on the first item in the side bar Dashboard",
    )

def test_dashboard_roles():
    """Test which roles can access dashboard"""
    assert query_and_validate(
        question="Which roles can access the dashboard?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_dashboard_location():
    """Test where dashboard is located"""
    assert query_and_validate(
        question="Where is the dashboard located?",
        expected_response="Dashboard Page",
    )

def test_dashboard_use_case():
    """Test dashboard use case"""
    assert query_and_validate(
        question="When should you use the dashboard?",
        expected_response="When the user wants to Track the flow of work without accessing each item",
    )

def test_dashboard_content():
    """Test dashboard content"""
    assert query_and_validate(
        question="What does the dashboard contain?",
        expected_response="Dashboard contains: Designs Needed by You Tiling Lead/Job, Total Jobs, Pending Approval, User Types, Lead Types, Completed & Cancelled Jobs, Acre Types, Bookkeeping, Shared with Designers, Total Equipment, in Maintenance, Nearing Maintenance",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAP VIEW TESTS (DOC23)
# ═══════════════════════════════════════════════════════════════════════════════

def test_map_view_access():
    """Test how to access map view"""
    assert query_and_validate(
        question="How do you access the map view?",
        expected_response="Click on the first map icon in the upper bar Map View",
    )

def test_map_view_roles():
    """Test which roles can access map view"""
    assert query_and_validate(
        question="Which roles can access the map view?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_map_view_location():
    """Test where map view is located"""
    assert query_and_validate(
        question="Where is the map view located?",
        expected_response="Map View Page",
    )

def test_map_view_use_case():
    """Test map view use case"""
    assert query_and_validate(
        question="When should you use the map view?",
        expected_response="When the user wants to Track the flow of work on the map without accessing each item",
    )

def test_map_view_content():
    """Test map view content"""
    assert query_and_validate(
        question="What does the map view contain?",
        expected_response="Map View contains: Repair Leads, Excavation Leads, Tiling Leads, Repair Jobs, Excavation Jobs, and Tiling Jobs on the map, with a designated icon for each",
    )

def test_map_view_filtering():
    """Test map view filtering options"""
    assert query_and_validate(
        question="How can you filter the map view?",
        expected_response="You can filter the view by pressing on Select Filters and choosing the desired filter: All Jobs, All Leads and Jobs, All Leads, All Jobs, Excavation Jobs, Excavation Leads, Repair Jobs, Repair Leads, Tiling Jobs, Tiling Leads",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CONTACT INFO TESTS (DOC24)
# ═══════════════════════════════════════════════════════════════════════════════

def test_contact_info_access_steps():
    """Test steps to access contact info"""
    assert query_and_validate(
        question="How do you access contact info?",
        expected_response="Click on the upper right corner logo, click on Contact Info, now you can view or add contacts",
    )

def test_contact_info_roles():
    """Test which roles can access contact info"""
    assert query_and_validate(
        question="Which roles can access contact info?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_contact_info_location():
    """Test where contact info is located"""
    assert query_and_validate(
        question="Where is contact info located?",
        expected_response="Contact Info Page",
    )

def test_contact_info_add_steps():
    """Test steps to add contacts"""
    assert query_and_validate(
        question="How do you add contacts?",
        expected_response="Press on Add Contact button, choose between client contact or general contact, fill full name phone number email address and Description then press Add contact to create the contact",
    )

def test_contact_info_contact_types():
    """Test types of contacts that can be added"""
    assert query_and_validate(
        question="What types of contacts can be added?",
        expected_response="Client contact or general contact",
    )

def test_contact_info_required_fields():
    """Test required fields for adding contacts"""
    assert query_and_validate(
        question="What information is needed to add a contact?",
        expected_response="Full name, phone number, email address, and Description",
    )

def test_contact_info_management_permissions():
    """Test contact management permissions"""
    assert query_and_validate(
        question="Who can add, edit, or delete contacts?",
        expected_response="Contacts can be added, edited, or deleted only by admins",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLED FOOTAGE ANALYTICS TESTS (DOC25)
# ═══════════════════════════════════════════════════════════════════════════════

def test_installed_footage_access():
    """Test how to access installed footage analytics"""
    assert query_and_validate(
        question="How do you access installed footage analytics?",
        expected_response="Click on the second item in the side bar Installed footage, click on Installed Footage",
    )

def test_installed_footage_roles():
    """Test which roles can access installed footage analytics"""
    assert query_and_validate(
        question="Which roles can access installed footage analytics?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_installed_footage_location():
    """Test where installed footage analytics is located"""
    assert query_and_validate(
        question="Where is installed footage analytics located?",
        expected_response="Installed Footage Page",
    )

def test_installed_footage_use_case():
    """Test installed footage analytics use case"""
    assert query_and_validate(
        question="When should you use installed footage analytics?",
        expected_response="When the user wants to Track the flow of installed footage without accessing each item",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZATION SETTINGS TESTS (DOC26)
# ═══════════════════════════════════════════════════════════════════════════════

def test_organization_settings_access():
    """Test how to access organization settings"""
    assert query_and_validate(
        question="How do you access organization settings?",
        expected_response="Click on the logo on the upper left side, click on Settings, now you can view or modify the organization settings",
    )

def test_organization_settings_roles():
    """Test which roles can access organization settings"""
    assert query_and_validate(
        question="Which roles can access organization settings?",
        expected_response="Admin, Project Manager, Project Crew, Book Keeper, and Viewer",
    )

def test_organization_settings_location():
    """Test where organization settings is located"""
    assert query_and_validate(
        question="Where are organization settings located?",
        expected_response="Organization Settings Page",
    )

def test_organization_settings_modification():
    """Test organization settings modification capabilities"""
    assert query_and_validate(
        question="What can be modified in organization settings?",
        expected_response="By pressing add status, the user can add a job or leads status, and by pressing add type, the user can also add a lead type. System setting numbers can also be edited by removing old number and adding a new number",
    )

def test_organization_settings_job_statuses():
    """Test job status settings"""
    assert query_and_validate(
        question="What are the default job statuses and how can they be customized?",
        expected_response="For repair and excavation jobs, the default statuses are New, In progress, and Completed. For tiling jobs, the default statuses are New, Location, In Progress-Installation, In Progress-Clean Up, Completed. Statuses can be added between new and completed by assigning the status number after pressing add status",
    )

def test_organization_settings_system_settings():
    """Test system settings configuration"""
    assert query_and_validate(
        question="What system settings can be edited?",
        expected_response="Equipment Maintenance Reminder, Equipment Nearing Maintenance, Automatically Archive After times can be edited",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM SETTINGS TESTS (DOC26 SYSTEM SETTINGS)
# ═══════════════════════════════════════════════════════════════════════════════

def test_system_settings_access():
    """Test how to access system settings"""
    assert query_and_validate(
        question="How do you access system settings?",
        expected_response="Click on the logo on the upper left side, click on Settings, now you can view or modify the organization settings",
    )

def test_system_settings_roles():
    """Test which roles can access system settings"""
    assert query_and_validate(
        question="Which roles can access system settings?",
        expected_response="Admin and Project Manager",
    )

def test_system_settings_job_status_configuration():
    """Test job status configuration in system settings"""
    assert query_and_validate(
        question="How can job statuses be configured in system settings?",
        expected_response="Job status can be for repair, excavation, and tiling. For repair and excavation jobs, the default statuses are New, In progress, and Completed. For tiling jobs, the default statuses are New, Location, In Progress-Installation, In Progress-Clean Up, Completed. Statuses can be added between new and completed by assigning the status number after pressing add status",
    )

def test_system_settings_maintenance_configuration():
    """Test maintenance settings configuration"""
    assert query_and_validate(
        question="What maintenance-related settings can be configured?",
        expected_response="Equipment Maintenance Reminder, Equipment Nearing Maintenance, Automatically Archive After times can be edited",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL SYSTEM FEATURES TESTS (DOC27)
# ═══════════════════════════════════════════════════════════════════════════════

def test_general_system_sections():
    """Test general system sections description"""
    assert query_and_validate(
        question="What are the main sections of the system?",
        expected_response="Leads (Repair-Excavation-Tiling): Potential projects or client inquiries. Jobs (Repair-Excavation-Tiling): Confirmed and active work orders. Bookkeeping: Financial tracking section. Order Pipes: Management of material requests. Dashboard: Centralized overview. Maintenance: Equipment condition monitoring. Knowledge Base: Support and help resources",
    )

def test_general_leads_po_format():
    """Test leads PO number format"""
    assert query_and_validate(
        question="What is the format of PO numbers for leads?",
        expected_response="The PO Number consists of the year of creation, order of creation, and the last letters are abbreviations for: RPL(Repair Lead), EXL (Excavation Lead), TLL(tiling lead)",
    )

def test_general_jobs_po_format():
    """Test jobs PO number format"""
    assert query_and_validate(
        question="What is the format of PO numbers for jobs?",
        expected_response="The PO Number consists of the year of creation, order of creation, and the last letters are abbreviations for: RP(Repair job), EX (Excavation job), TL(tiling job)",
    )

def test_general_job_progress():
    """Test job progress calculation"""
    assert query_and_validate(
        question="How is job progress calculated?",
        expected_response="Job progress changes with the number of the status that a job is currently in",
    )

def test_general_equipment_filters():
    """Test equipment filtering options"""
    assert query_and_validate(
        question="What filtering options are available for equipment?",
        expected_response="In machines, we have type (machines, vehicles, trailers), status (available/unavailable), and filters for (machines, vehicles, trailers)",
    )

def test_general_delete_unarchive_requirement():
    """Test deletion requirements for archived items"""
    assert query_and_validate(
        question="What is required to delete archived jobs or leads?",
        expected_response="To delete a job or a lead, they must be unarchived (in case they were archived)",
    )

def test_general_unarchive_process():
    """Test unarchiving process"""
    assert query_and_validate(
        question="How do you unarchive a job or lead?",
        expected_response="To unarchive a job or a lead, in the archived section, press on the three points on the left, then unarchive. The job will be moved back to the active section",
    )

def test_general_811_call_requirement():
    """Test 811 call location requirement"""
    assert query_and_validate(
        question="What is required to perform an 811 call?",
        expected_response="To perform the 811 call in a lead or a job, it is required to add the location of the job",
    )

def test_general_notes_comments_features():
    """Test notes and comments system features"""
    assert query_and_validate(
        question="What features are available for notes and comments?",
        expected_response="All notes and comments in the system can be edited, deleted, and other members can be mentioned",
    )


def test_general_knowledge_restriction():
    """Test general knowledge question handling"""
    assert query_and_validate(
        question="What is the capital of France?",
        expected_response="This question is outside the scope of the RAG system",
    )

def test_use_case_scenarios():
    """Test use case scenario responses"""
    assert query_and_validate(
        question="Can you provide a use case scenario for a repair lead?",
        expected_response="A use case scenario for a repair lead involves a customer contacting the company for a repair, the company creating a repair lead with the customer's information and the details of the repair needed, and then converting that lead into a job once it's been approved.",
    )

def test_conversion_trigger_scenario():
    """Test the scenario that triggers conversion of a lead to a job"""
    assert query_and_validate(
        question="What triggers the conversion of a lead to a job?",
        expected_response="The conversion of a lead to a job is triggered when the necessary information is provided and the lead is approved by the admin or project manager.",
    )

def test_job_creation_scenario():
    """Test the scenario of creating a job from a lead"""
    assert query_and_validate(
        question="Can you describe the job creation process from a lead?",
        expected_response="The job creation process from a lead involves reviewing the lead details, assigning a designer and equipment if necessary, and then clicking on the 'Convert to job' button to create the job.",
    )


def run_all_tests():
    """Run all comprehensive test cases"""
    tests = [
     
        # Repair Lead Tests
        test_add_repair_lead_steps,
        test_add_repair_lead_roles,
        test_repair_lead_required_fields,
        test_repair_lead_optional_fields,
        test_repair_lead_equipment_restriction,
        test_repair_lead_designer_restriction,
        test_repair_lead_access_methods,
        test_repair_lead_archiving,
        test_repair_lead_deletion_policy,
        test_repair_lead_811_call,
        test_repair_lead_status_changes,
        test_repair_lead_notes_comments,
        test_repair_lead_file_management,
        test_repair_lead_customer_info_editing,
        test_repair_lead_location_editing,
        
        # Excavation Lead Tests
        test_add_excavation_lead_steps,
        test_add_excavation_lead_roles,
        test_excavation_lead_equipment_designer_restriction,
        test_excavation_lead_designer_restriction,
        test_excavation_lead_status_range,
        
        # Tiling Lead Tests
        test_add_tiling_lead_steps,
        test_add_tiling_lead_roles,
        test_tiling_lead_equipment_designer_restriction,
        test_tiling_lead_designer_restriction,
        
        # Lead Conversion Tests
        test_convert_repair_lead_requirements,
        test_convert_repair_lead_steps,
        test_convert_tiling_lead_requirements,
        test_convert_tiling_lead_steps,
        test_convert_excavation_lead_requirements,
        test_convert_excavation_lead_steps,
        test_lead_conversion_data_transfer,
        test_job_reversion_restriction,
        
        # Job Creation Tests
        test_add_repair_job_steps,
        test_add_repair_job_roles,
        test_repair_job_required_fields,
        test_repair_job_equipment_designer_restriction,
        test_repair_job_designer_restriction,
        test_add_excavation_job_steps,
        test_excavation_job_equipment_capability,
        test_excavation_job_designer_restriction,
        test_excavation_job_hours_setting,
        test_add_tiling_job_steps,
        test_tiling_job_machines_capability,
        test_tiling_job_designer_capability,
        test_tiling_job_project_crew,
        test_tiling_job_vehicle_restriction,
        test_tiling_job_installed_footage,
        
        # Business Features Tests
        test_repair_job_invoicing,
        test_excavation_job_invoicing,
        test_tiling_job_invoicing,
        test_tiling_job_pipe_ordering,
        
        # Invoice Creation Tests (DOC10)
        test_invoice_creation_steps,
        test_invoice_creation_roles,
        test_invoice_creation_location,
        test_invoice_job_types,
        test_invoice_client_info_fields,
        test_invoice_item_fields,
        test_invoice_admin_check_requirement,
        test_invoice_admin_check_restriction,
        test_invoice_sent_to_client_workflow,
        test_invoice_payment_workflow,
        test_invoice_multiple_per_job,
        test_invoice_multiple_orders,
        test_invoice_deletion,
        
        # Order Pipes Tests (DOC11)
        test_order_pipes_steps,
        test_order_pipes_roles,
        
    ]
    
    passed_tests = 0
    failed_tests = 0
    total_tests = len(tests)
    
    print(f"\n🧪 Running {total_tests} comprehensive test cases...")
    print("=" * 80)
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{total_tests}] Running {test_func.__name__}...")
        try:
            test_func()
            passed_tests += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed_tests += 1
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed.")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)