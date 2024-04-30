import { Component, OnInit } from '@angular/core';
import {FormBuilder, FormGroup} from '@angular/forms';
import { EmployeeModel } from './employee-dashboard.model';
import { ApiService } from '../shared/api.service';
@Component({
  selector: 'app-employee-dashboard',
  templateUrl: './employee-dashboard.component.html',
  styleUrls: ['./employee-dashboard.component.css']
})
export class EmployeeDashboardComponent implements OnInit{
  [x: string]: any;

  formvalue !: FormGroup;
  employeeModelObj : EmployeeModel = new EmployeeModel();
  constructor(private frombuilder: FormBuilder, private api: ApiService){}

  ngOnInit(): void {
  this.formvalue = this.frombuilder.group({
    firstName : [''],
    lastName : [''],
    email : [''],
    mobile : [''],
    salary : [''],
  })
  }

  postEmployeeDetails(){
  this.employeeModelObj.firstName = this.formvalue.value.firstName;
  this.employeeModelObj.lastName = this.formvalue.value.lastName;
  this.employeeModelObj.email = this.formvalue.value.email;
  this.employeeModelObj.mobile = this.formvalue.value.mobile;
  this.employeeModelObj.salary = this.formvalue.value.salary;

 this.api.postEmployee(this.employeeModelObj).subscribe(res=>{
    console.log(res);
    alert("Employee Added Successfully")
    this.formvalue.reset();
  });

    (    error: any)=>{
    alert("Something Went Wrong")
  };

  }

}
