// <auto-generated/>
#pragma warning disable 1591
#pragma warning disable 0414
#pragma warning disable 0649
#pragma warning disable 0169

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
    public partial class ModifyModalForm___복사 : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
        }
        #pragma warning restore 1998
#nullable restore
#line 21 "D:\Kipi\Clothink\Clothink\Client\Pages\ModifyModalForm - 복사.razor"
       
    [Parameter]
    public RenderFragment ChildContent { get; set; }

    [Parameter]
    public string Title { get; set; }

    [Parameter]
    public bool ShowModal { get; set; }

    [Parameter]
    public EventCallback<bool> ShowModalChanged { get; set; }

    private async Task CloseModal()
    {
        ShowModal = false;
        await ShowModalChanged.InvokeAsync(false);
    }

#line default
#line hidden
#nullable disable
    }
}
#pragma warning restore 1591
