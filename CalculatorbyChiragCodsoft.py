import tkinter as t
def KeyPressed(i):
    if i!='=':
        entry.insert(t.END,i)
    else:
        expression=entry.get()
        try:
            value=eval(expression)
            new_string=f'{value:,}'
            entry.delete(0,t.END)
            entry.insert(0,new_string)
        except Exception as e:
            entry.delete(0,t.END)
            entry.insert(0,e)

root=t.Tk()
root.title("Calculator by Chirag")

entry=t.Entry(root,width=30,borderwidth=5,font=('arial',15))
entry.grid(row=0,column=0,padx=10,pady=10,columnspan=5)
btn_frame=t.Frame(root,width=40,height=180)
btn_frame.grid(row=1,columnspan=5)
Key_Text=['7','8','9','*','4','5','6','-','1','2','3','+','/','0','.','=']
i=j=0
for x in Key_Text:
    b=t.Button(btn_frame,width=5,text=x,command=lambda x = x: KeyPressed(x))
    b.grid(row=i,column=j,ipadx=2,ipady=4)
    j+=1
    if j==4:
        i+=1
        j=0
del_btn=t.Button(root,text='Reset',width=30,command=lambda:entry.delete(0,t.END))
del_btn.grid(row=2,columnspan=5,ipadx=5,ipady=4)
root.mainloop()