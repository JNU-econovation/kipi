#pragma checksum "D:\Kipi\Clothink\Clothink\Client\Pages\ButtonWithModifyModal.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "11887f8546fad0e6da55e46c1901c38914897ea0"
// <auto-generated/>
#pragma warning disable 1591
namespace Clothink.Client.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using System.Net.Http.Json;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.AspNetCore.Components.Web.Virtualization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.AspNetCore.Components.WebAssembly.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Clothink.Client;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "D:\Kipi\Clothink\Clothink\Client\_Imports.razor"
using Clothink.Client.Shared;

#line default
#line hidden
#nullable disable
    public partial class ButtonWithModifyModal : global::Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.OpenElement(0, "button");
            __builder.AddAttribute(1, "class", "btn btn-primary");
            __builder.AddAttribute(2, "onclick", global::Microsoft.AspNetCore.Components.EventCallback.Factory.Create<global::Microsoft.AspNetCore.Components.Web.MouseEventArgs>(this, 
#nullable restore
#line 2 "D:\Kipi\Clothink\Clothink\Client\Pages\ButtonWithModifyModal.razor"
                                          () => ShowModal = true

#line default
#line hidden
#nullable disable
            ));
            __builder.AddMarkupContent(3, "수정");
            __builder.CloseElement();
            __builder.AddMarkupContent(4, ">\r\n");
            __builder.OpenComponent<global::Clothink.Client.Pages.ModifyModalForm>(5);
            __builder.AddAttribute(6, "Title", (object)("My Modal"));
            __builder.AddAttribute(7, "ShowModal", (object)(global::Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.TypeCheck<global::System.Boolean>(
#nullable restore
#line 3 "D:\Kipi\Clothink\Clothink\Client\Pages\ButtonWithModifyModal.razor"
                                              ShowModal

#line default
#line hidden
#nullable disable
            )));
            __builder.AddAttribute(8, "ShowModalChanged", (object)(global::Microsoft.AspNetCore.Components.CompilerServices.RuntimeHelpers.TypeCheck<global::Microsoft.AspNetCore.Components.EventCallback<global::System.Boolean>>(global::Microsoft.AspNetCore.Components.EventCallback.Factory.Create<global::System.Boolean>(this, 
#nullable restore
#line 3 "D:\Kipi\Clothink\Clothink\Client\Pages\ButtonWithModifyModal.razor"
                                                                           (value) => ShowModal = value

#line default
#line hidden
#nullable disable
            ))));
            __builder.AddAttribute(9, "ChildContent", (global::Microsoft.AspNetCore.Components.RenderFragment)((__builder2) => {
                __builder2.AddMarkupContent(10, "<p>This is the content of the modal.</p>");
            }
            ));
            __builder.CloseComponent();
        }
        #pragma warning restore 1998
#nullable restore
#line 7 "D:\Kipi\Clothink\Clothink\Client\Pages\ButtonWithModifyModal.razor"
       
    private bool ShowModal = false;

#line default
#line hidden
#nullable disable
    }
}
#pragma warning restore 1591
